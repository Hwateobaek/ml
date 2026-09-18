/**
 * `mlpx-gradient-boosting-v1`·`mlpx-gradient-boosting-regression-v1` — 그레이디언트 부스팅
 * (mlpx-spec.md §5.13).
 *
 * **나무를 앞 나무가 틀린 만큼씩 쌓는다.** 예측은 시작값(`init`)에 나무마다의 값을
 * 학습률(`learningRate`)만큼 줄여 더한 점수다 — sklearn `GradientBoostingClassifier`·
 * `GradientBoostingRegressor`의 `decision_function`과 같은 식이다.
 *
 * **분류와 회귀가 형식을 가른다.** 돌려주는 것이 라벨이냐 수치냐가 다르고 분류만 확률이
 * 있다 — 인공신경망이 형식을 가른 것과 같은 판단이다 (mlpx-spec.md §5.11).
 *
 * **분류 점수의 칸 수:**
 * - 이진이면 **한 칸**이다. 점수가 `log(p/(1−p))`이고 0보다 크면 `classes[1]`이다.
 * - 다중이면 **클래스 수만큼**이고 가장 큰 칸이 답이다(softmax). 동점이면 앞 칸이 이긴다.
 * sklearn이 이진과 다중을 이렇게 가르고, 그래서 여기도 같다.
 *
 * **나무 한 그루를 걷는 규칙은 `regression-tree.ts`의 것이다.** 두 벌이면 저장 전후가
 * 한쪽에서만 갈린다.
 */

import { z } from 'zod'

import type { ModelFile, Predict, ProbaModel } from './types'
import {
  compileRegressionTree,
  invalidModel,
  regressionNodeSchema,
  walkRegressionTree,
  type CompiledRegressionTree,
  type RegressionNode,
} from './regression-tree'

export const GRADIENT_BOOSTING_FORMAT = 'mlpx-gradient-boosting-v1'
export const GRADIENT_BOOSTING_REGRESSION_FORMAT = 'mlpx-gradient-boosting-regression-v1'

type Nodes = readonly RegressionNode[]

export interface GradientBoostingModel extends ModelFile {
  readonly format: typeof GRADIENT_BOOSTING_FORMAT
  /** 라벨을 **정렬한** 순서. 이진이면 점수가 `classes[1]` 쪽이다. */
  readonly classes: readonly string[]
  readonly featureCount: number
  readonly learningRate: number
  /** 점수 칸마다의 시작값. 이진이면 한 칸, 다중이면 클래스 수만큼이다. */
  readonly init: readonly number[]
  /** 단계마다 점수 칸 수만큼의 나무. `stages[m][c]`가 m번째 단계의 c번 칸이다. */
  readonly stages: readonly (readonly { readonly nodes: Nodes }[])[]
}

export interface GradientBoostingRegressionModel extends ModelFile {
  readonly format: typeof GRADIENT_BOOSTING_REGRESSION_FORMAT
  readonly featureCount: number
  readonly learningRate: number
  /** 시작값. 훈련 타깃의 평균이다 (sklearn의 `init='zero'`가 아닌 기본값). */
  readonly init: number
  readonly stages: readonly { readonly nodes: Nodes }[]
}

const treeSchema = z.looseObject({ nodes: z.array(regressionNodeSchema).min(1) })

const classificationSchema = z.looseObject({
  format: z.literal(GRADIENT_BOOSTING_FORMAT),
  classes: z.array(z.string()).min(2),
  featureCount: z.number(),
  learningRate: z.number(),
  init: z.array(z.number()).min(1),
  stages: z.array(z.array(treeSchema).min(1)),
})

const regressionSchema = z.looseObject({
  format: z.literal(GRADIENT_BOOSTING_REGRESSION_FORMAT),
  featureCount: z.number(),
  learningRate: z.number(),
  init: z.number(),
  stages: z.array(treeSchema),
})

/** 이진은 한 칸, 다중은 클래스 수만큼. 그 밖의 칸 수는 깨진 파일이다. */
export function scoreWidth(classCount: number): number {
  return classCount === 2 ? 1 : classCount
}

function checkCommon(featureCount: number, learningRate: number): void {
  if (!Number.isInteger(featureCount) || featureCount <= 0) invalidModel('featureCount')
  if (!Number.isFinite(learningRate) || learningRate <= 0) invalidModel('learningRate')
}

/** 점수 계산기. 분류 예측과 확률이 **같은 점수를 쓴다** — 라벨과 확률이 갈리지 않게. */
function classificationScorer(
  model: GradientBoostingModel,
): (row: readonly number[]) => Float64Array {
  const { classes, featureCount, learningRate, init, stages } = model
  checkCommon(featureCount, learningRate)
  const width = scoreWidth(classes.length)
  if (init.length !== width) invalidModel('init')
  if (!init.every((value) => Number.isFinite(value))) invalidModel('init')
  const compiled: CompiledRegressionTree[][] = stages.map((stage) => {
    if (stage.length !== width) invalidModel('stages')
    return stage.map((tree) => compileRegressionTree(tree.nodes, featureCount))
  })

  return (row) => {
    if (row.length !== featureCount) invalidModel('featureCount')
    const scores = Float64Array.from(init)
    for (const stage of compiled) {
      for (let c = 0; c < width; c += 1) {
        scores[c] = (scores[c] ?? 0) + learningRate * walkRegressionTree(stage[c]!, row)
      }
    }
    return scores
  }
}

/** 점수에서 라벨 번호. 이진은 부호로, 다중은 가장 큰 칸으로 — 동점이면 앞 칸이 이긴다. */
export function classIndexOf(scores: Float64Array): number {
  if (scores.length === 1) return (scores[0] ?? 0) > 0 ? 1 : 0
  let best = 0
  for (let c = 1; c < scores.length; c += 1) {
    if ((scores[c] ?? 0) > (scores[best] ?? 0)) best = c
  }
  return best
}

/** 점수에서 확률. 이진은 로지스틱, 다중은 softmax(최댓값을 빼서 넘침을 막는다). */
export function probabilitiesOf(scores: Float64Array): Float64Array {
  if (scores.length === 1) {
    const p = 1 / (1 + Math.exp(-(scores[0] ?? 0)))
    return Float64Array.from([1 - p, p])
  }
  let max = Number.NEGATIVE_INFINITY
  for (const score of scores) if (score > max) max = score
  const out = new Float64Array(scores.length)
  let sum = 0
  for (let c = 0; c < scores.length; c += 1) {
    const value = Math.exp((scores[c] ?? 0) - max)
    out[c] = value
    sum += value
  }
  for (let c = 0; c < out.length; c += 1) out[c] = (out[c] ?? 0) / sum
  return out
}

/** 분류 예측. **학습 쪽도 이 함수를 그대로 쓴다.** */
export function gradientBoostingPredict(model: GradientBoostingModel): Predict {
  const score = classificationScorer(model)
  return (features) =>
    features.map((row) => {
      const label = model.classes[classIndexOf(score(row))]
      if (label === undefined) invalidModel('classes')
      return label
    })
}

/** 분류 확률. 칸 순서는 `classes`와 같다 (mlpx-spec.md §5.4). */
export function gradientBoostingProba(model: GradientBoostingModel): ProbaModel {
  const score = classificationScorer(model)
  return {
    classes: model.classes,
    predict: (features) => features.map((row) => probabilitiesOf(score(row))),
  }
}

/** 회귀 예측. **학습 쪽도 이 함수를 그대로 쓴다.** */
export function gradientBoostingRegressionPredict(model: GradientBoostingRegressionModel): Predict {
  const { featureCount, learningRate, init, stages } = model
  checkCommon(featureCount, learningRate)
  if (!Number.isFinite(init)) invalidModel('init')
  const compiled = stages.map((tree) => compileRegressionTree(tree.nodes, featureCount))
  return (features) =>
    features.map((row) => {
      if (row.length !== featureCount) invalidModel('featureCount')
      let score = init
      for (const tree of compiled) score += learningRate * walkRegressionTree(tree, row)
      return score
    })
}

function parseClassification(file: unknown): GradientBoostingModel {
  const parsed = classificationSchema.safeParse(file)
  if (!parsed.success) invalidModel('payload')
  return { ...parsed.data, format: GRADIENT_BOOSTING_FORMAT }
}

export function loadGradientBoostingModel(file: unknown): Predict {
  return gradientBoostingPredict(parseClassification(file))
}

export function loadGradientBoostingProba(file: unknown): ProbaModel {
  return gradientBoostingProba(parseClassification(file))
}

export function loadGradientBoostingRegressionModel(file: unknown): Predict {
  const parsed = regressionSchema.safeParse(file)
  if (!parsed.success) invalidModel('payload')
  return gradientBoostingRegressionPredict({
    ...parsed.data,
    format: GRADIENT_BOOSTING_REGRESSION_FORMAT,
  })
}
