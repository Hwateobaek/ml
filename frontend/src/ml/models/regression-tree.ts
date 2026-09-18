/**
 * `mlpx-tree-regression-v1` — 결정트리 회귀와 랜덤 포레스트 회귀 (mlpx-spec.md §5.12).
 *
 * **분류 나무(`mlpx-tree-v1`)와 형식이 갈린다.** 잎이 담는 것이 클래스 번호가 아니라
 * **수치**이고, 여러 그루를 묶는 규칙이 다수결이 아니라 **평균**이다. `classes`를 선택
 * 필드로 두고 해석기가 그 유무로 갈라지면 한 형식이 payload 둘을 갖게 된다 — 인공신경망이
 * 회귀 형식을 따로 둔 것과 같은 판단이다 (mlpx-spec.md §5.11).
 *
 * **결정트리는 나무가 한 그루인 포레스트다** — 분류 형식과 같다. 한 그루의 평균은 그
 * 나무의 답이다.
 *
 * **나누는 규칙은 `x <= 임계값`이면 왼쪽이다.** sklearn의 `tree_.threshold`와 같은 방향이고
 * 분류 형식(`<`)과 다르다 — 이 형식은 우리 나무(`engines/cart-regression.ts`)가 쓰고,
 * 임계값이 이웃한 두 값의 가운데라 훈련 데이터에서는 둘이 같은 답을 낸다.
 *
 * **그레이디언트 부스팅도 이 파일의 나무를 쓴다** (`gradient-boosting.ts`). 나무 한 그루를
 * 걷는 규칙이 두 벌이면 저장 전후의 예측이 한쪽에서만 갈린다.
 */

import { z } from 'zod'

import { ClientError, type ClientErrorParams } from '../../errors'
import type { ModelFile, Predict } from './types'

export const TREE_REGRESSION_FORMAT = 'mlpx-tree-regression-v1'

/** 잎 표시. 분류 형식과 같은 값이다 — 자식 자리까지 채워 검증할 수 있게 한다. */
export const REGRESSION_LEAF = -1

/** `[열, 임계값, 왼쪽, 오른쪽]`. 잎이면 `[-1, 값, -1, -1]`. */
export type RegressionNode = readonly [number, number, number, number]

export interface RegressionTreeModel extends ModelFile {
  readonly format: typeof TREE_REGRESSION_FORMAT
  /** 전처리를 마친 행렬의 열 수. 이 값과 안 맞는 입력은 거부한다. */
  readonly featureCount: number
  readonly trees: readonly { readonly nodes: readonly RegressionNode[] }[]
}

export const regressionNodeSchema = z.tuple([z.number(), z.number(), z.number(), z.number()])

const regressionTreeSchema = z.looseObject({
  format: z.literal(TREE_REGRESSION_FORMAT),
  featureCount: z.number(),
  trees: z.array(z.looseObject({ nodes: z.array(regressionNodeSchema).min(1) })).min(1),
})

/** 예측 전용 표현. 분류 형식(`tree.ts`)과 같은 이유로 TypedArray로 편다. */
export interface CompiledRegressionTree {
  readonly column: Int32Array
  readonly value: Float64Array
  readonly left: Int32Array
  readonly right: Int32Array
}

export function invalidModel(field: string): never {
  throw new ClientError('MODEL_FILE_INVALID', { field } satisfies ClientErrorParams)
}

function isIndex(value: number, limit: number): boolean {
  return Number.isInteger(value) && value >= 0 && value < limit
}

/**
 * 검증하면서 예측 전용 표현으로 옮긴다.
 *
 * **자식은 반드시 자기보다 뒤에 있어야 한다** — 분류 형식과 같은 조건이고, 이것 하나가
 * 순환을 구조적으로 막아 `walkRegressionTree`의 루프가 반드시 끝난다.
 */
export function compileRegressionTree(
  nodes: readonly RegressionNode[],
  featureCount: number,
): CompiledRegressionTree {
  const size = nodes.length
  const column = new Int32Array(size)
  const value = new Float64Array(size)
  const left = new Int32Array(size)
  const right = new Int32Array(size)

  nodes.forEach((node, index) => {
    const [nodeColumn, nodeValue, nodeLeft, nodeRight] = node
    if (!Number.isFinite(nodeValue)) invalidModel('value')
    if (nodeColumn === REGRESSION_LEAF) {
      if (nodeLeft !== REGRESSION_LEAF || nodeRight !== REGRESSION_LEAF) invalidModel('leafChild')
    } else {
      if (!isIndex(nodeColumn, featureCount)) invalidModel('column')
      if (!isIndex(nodeLeft, size) || nodeLeft <= index) invalidModel('left')
      if (!isIndex(nodeRight, size) || nodeRight <= index) invalidModel('right')
    }
    column[index] = nodeColumn
    value[index] = nodeValue
    left[index] = nodeLeft
    right[index] = nodeRight
  })

  return { column, value, left, right }
}

/** 나무 한 그루가 내는 값. 자식 인덱스가 단조 증가하므로 루프는 반드시 끝난다. */
export function walkRegressionTree(tree: CompiledRegressionTree, row: ArrayLike<number>): number {
  let index = 0
  for (;;) {
    const column = tree.column[index]
    const value = tree.value[index]
    if (column === undefined || value === undefined) invalidModel('node')
    if (column === REGRESSION_LEAF) return value

    const feature = row[column]
    const left = tree.left[index]
    const right = tree.right[index]
    if (feature === undefined || left === undefined || right === undefined) invalidModel('node')
    index = feature <= value ? left : right
  }
}

/**
 * 나무들의 평균을 예측 함수로. **학습 쪽도 이 함수를 그대로 쓴다** — KNN·SVM과 같은
 * 방식이고, 그래서 저장했다 읽은 모델의 예측이 원본과 같은 것이 구조로 보장된다.
 */
export function regressionForestPredict(model: RegressionTreeModel): Predict {
  const { featureCount } = model
  if (!Number.isInteger(featureCount) || featureCount <= 0) invalidModel('featureCount')
  const trees = model.trees.map((tree) => compileRegressionTree(tree.nodes, featureCount))
  if (trees.length === 0) invalidModel('trees')

  return (features) =>
    features.map((row) => {
      // 폭이 다른 입력은 다른 열로 예측한다 — 실패가 아니라 조용히 틀린 숫자다.
      if (row.length !== featureCount) invalidModel('featureCount')
      let sum = 0
      for (const tree of trees) sum += walkRegressionTree(tree, row)
      return sum / trees.length
    })
}

/** 파일 내용을 예측 함수로. 형식과 안 맞으면 던진다. */
export function loadRegressionTreeModel(file: unknown): Predict {
  const parsed = regressionTreeSchema.safeParse(file)
  if (!parsed.success) invalidModel('payload')
  return regressionForestPredict({
    format: TREE_REGRESSION_FORMAT,
    featureCount: parsed.data.featureCount,
    trees: parsed.data.trees,
  })
}
