/**
 * 그레이디언트 부스팅 — **앞 나무가 틀린 만큼을 다음 나무가 배운다** (mlpx-spec.md §5.13).
 *
 * 랜덤 포레스트는 나무를 **따로** 키워 평균하고(배깅), 여기는 나무를 **차례로** 쌓는다.
 * 나무마다 지금까지의 예측이 틀린 방향(손실의 음의 기울기)을 배우고, 그 답을
 * 학습률만큼 줄여 더한다. 두 앙상블을 나란히 견주는 것이 이 알고리즘을 넣은 이유다.
 *
 * **sklearn `GradientBoostingClassifier`·`GradientBoostingRegressor` 기본값의 식을
 * 따른다** — 손실(회귀는 제곱오차, 분류는 로그 손실), 시작값, 잎의 값을 정하는 뉴턴 한
 * 걸음까지. 손잡이는 셋이다(`n_estimators`·`learning_rate`·`max_depth`, 기본 100·0.1·3).
 * `subsample`은 1.0 고정이라 **난수를 안 쓴다** — 씨앗은 기록되지만 결과를 안 바꾼다.
 *
 * **sklearn과 갈릴 수 있는 곳.** 나무의 분할 기준이 sklearn의 `friedman_mse`가 아니라
 * 제곱오차이고(`cart-regression.ts`), 동점 분할을 앞 열이 가져간다. 동점이 없는
 * 데이터에서는 같은 분할을 고른다.
 */

import { GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT } from '../../limits'
import {
  GRADIENT_BOOSTING_FORMAT,
  GRADIENT_BOOSTING_REGRESSION_FORMAT,
  type GradientBoostingModel,
  type GradientBoostingRegressionModel,
  scoreWidth,
} from '../models/gradient-boosting'
import { walkRegressionTree, compileRegressionTree } from '../models/regression-tree'
import type { RegressionNode } from '../models/regression-tree'
import { allRows, growRegressionTree, toColumns } from './cart-regression'

export interface BoostingOptions {
  readonly nEstimators: number
  readonly learningRate: number
  readonly maxDepth: number
}

/** 뉴턴 한 걸음의 분모가 이보다 작으면 잎을 0으로 둔다 — sklearn과 같은 자리다. */
const TINY_DENOMINATOR = 1e-150

/** 나무 하나를 키우고, 훈련 행마다 그 나무의 값을 점수에 더한다. */
function growAndApply(
  columns: readonly Float64Array[],
  rowsOf: readonly (readonly number[])[],
  residuals: Float64Array,
  maxDepth: number,
  leafValue: ((rows: Int32Array) => number) | undefined,
  scores: Float64Array,
  learningRate: number,
  featureCount: number,
): RegressionNode[] {
  const nodes = growRegressionTree({
    columns,
    targets: residuals,
    rows: allRows(residuals.length),
    maxDepth,
    minSplit: GRADIENT_BOOSTING_MIN_SAMPLES_SPLIT,
    ...(leafValue ? { leafValue } : {}),
  })
  // **해석기의 걷기로 더한다** — 학습 중의 점수와 저장한 모델의 점수가 같은 규칙에서 나온다.
  const compiled = compileRegressionTree(nodes, featureCount)
  rowsOf.forEach((row, index) => {
    scores[index] = (scores[index] ?? 0) + learningRate * walkRegressionTree(compiled, row)
  })
  return nodes
}

/** 회귀. 시작값은 타깃의 평균이고, 나무마다 잔차(`y − 예측`)를 배운다. */
export function fitGradientBoostingRegression(
  features: readonly (readonly number[])[],
  targets: readonly number[],
  options: BoostingOptions,
): GradientBoostingRegressionModel {
  const featureCount = features[0]?.length ?? 0
  const columns = toColumns(features)
  const n = targets.length
  const init = n === 0 ? 0 : targets.reduce((sum, value) => sum + value, 0) / n
  const scores = new Float64Array(n).fill(init)
  const residuals = new Float64Array(n)
  const stages: { nodes: RegressionNode[] }[] = []

  for (let stage = 0; stage < options.nEstimators; stage += 1) {
    for (let i = 0; i < n; i += 1) residuals[i] = (targets[i] ?? 0) - (scores[i] ?? 0)
    const nodes = growAndApply(
      columns,
      features,
      residuals,
      options.maxDepth,
      undefined,
      scores,
      options.learningRate,
      featureCount,
    )
    stages.push({ nodes })
  }

  return {
    format: GRADIENT_BOOSTING_REGRESSION_FORMAT,
    featureCount,
    learningRate: options.learningRate,
    init,
    stages,
  }
}

/**
 * 분류. `encoded`는 정렬한 라벨의 번호이고 `classCount`는 2 이상이다.
 *
 * - **이진**: 점수 한 칸. 시작값이 `log(p/(1−p))`, 잎은 `Σr / Σp(1−p)`.
 * - **다중**: 점수 클래스 수만큼. 시작값이 `log(사전확률)`(칸끼리 평균을 빼 둔다 — softmax는
 *   모든 칸에 같은 수를 더해도 안 바뀐다), 잎은 `(K−1)/K · Σr / Σp(1−p)`.
 * 한 단계의 나무들은 **그 단계가 시작할 때의 점수**로 기울기를 잰다 — sklearn과 같다.
 */
export function fitGradientBoostingClassifier(
  features: readonly (readonly number[])[],
  encoded: readonly number[],
  classes: readonly string[],
  options: BoostingOptions,
): GradientBoostingModel {
  const featureCount = features[0]?.length ?? 0
  const columns = toColumns(features)
  const n = encoded.length
  const classCount = classes.length
  const width = scoreWidth(classCount)

  const counts = new Float64Array(classCount)
  for (const label of encoded) counts[label] = (counts[label] ?? 0) + 1
  const priors = Array.from(counts, (count) => count / Math.max(n, 1))

  let init: number[]
  if (width === 1) {
    const p = priors[1] ?? 0
    init = [Math.log(p / (1 - p))]
  } else {
    const logs = priors.map((p) => Math.log(p))
    const center = logs.reduce((sum, value) => sum + value, 0) / logs.length
    init = logs.map((value) => value - center)
  }

  // 점수는 칸 우선으로 편다: scores[c][i].
  const scores = init.map((value) => new Float64Array(n).fill(value))
  const stages: { nodes: RegressionNode[] }[][] = []
  const residuals = new Float64Array(n)
  const probabilities = init.map(() => new Float64Array(n))
  const scale = width === 1 ? 1 : (classCount - 1) / classCount

  for (let stage = 0; stage < options.nEstimators; stage += 1) {
    // 이 단계의 확률 — 단계 안의 나무들이 모두 이것을 쓴다.
    for (let i = 0; i < n; i += 1) {
      if (width === 1) {
        probabilities[0]![i] = 1 / (1 + Math.exp(-(scores[0]![i] ?? 0)))
      } else {
        let max = Number.NEGATIVE_INFINITY
        for (let c = 0; c < width; c += 1) max = Math.max(max, scores[c]![i] ?? 0)
        let sum = 0
        for (let c = 0; c < width; c += 1) {
          const value = Math.exp((scores[c]![i] ?? 0) - max)
          probabilities[c]![i] = value
          sum += value
        }
        for (let c = 0; c < width; c += 1) probabilities[c]![i] = (probabilities[c]![i] ?? 0) / sum
      }
    }

    const trees: { nodes: RegressionNode[] }[] = []
    for (let c = 0; c < width; c += 1) {
      // 이진이면 칸 하나가 클래스 1이다.
      const target = width === 1 ? 1 : c
      const p = probabilities[c]!
      for (let i = 0; i < n; i += 1) residuals[i] = (encoded[i] === target ? 1 : 0) - (p[i] ?? 0)

      const leafValue = (rows: Int32Array): number => {
        let numerator = 0
        let denominator = 0
        for (const row of rows) {
          numerator += residuals[row] ?? 0
          const q = p[row] ?? 0
          denominator += q * (1 - q)
        }
        return denominator < TINY_DENOMINATOR ? 0 : (scale * numerator) / denominator
      }

      // 잎 값을 정하는 동안 잔차가 바뀌면 안 된다 — 나무 하나가 끝날 때까지 이 배열을 안 건드린다.
      const nodes = growAndApply(
        columns,
        features,
        residuals,
        options.maxDepth,
        leafValue,
        scores[c]!,
        options.learningRate,
        featureCount,
      )
      trees.push({ nodes })
    }
    stages.push(trees)
  }

  return {
    format: GRADIENT_BOOSTING_FORMAT,
    classes: [...classes],
    featureCount,
    learningRate: options.learningRate,
    init,
    stages,
  }
}
