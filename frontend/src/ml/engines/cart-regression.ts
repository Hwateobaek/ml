/**
 * 회귀 나무를 키운다 — CART, 제곱오차 (mlpx-spec.md §5.12).
 *
 * **세 알고리즘이 이 함수 하나를 쓴다** — 결정트리 회귀, 랜덤 포레스트 회귀, 그레이디언트
 * 부스팅. 부스팅은 잎의 값만 자기 식으로 바꾼다(`leafValue`).
 *
 * **ml-cart의 회귀 나무를 안 쓰는 이유.** 그레이디언트 부스팅은 잎마다 자기 값을 계산해야
 * 하고(sklearn이 로그 손실에서 뉴턴 한 걸음을 잎에 넣는 자리), 남의 나무에는 그 입구가
 * 없다. 부스팅을 위해 나무를 하나 짜야 한다면 **회귀 셋이 같은 나무를 쓰는 편이** 규칙이
 * 한 벌이라 낫다.
 *
 * **규칙은 sklearn `DecisionTreeRegressor`와 같은 모양이다.**
 * - 나누는 기준은 제곱오차의 감소다(`criterion='squared_error'`).
 * - 임계값은 이웃한 두 값의 가운데이고, `x <= 임계값`이면 왼쪽이다.
 * - 뿌리의 깊이가 0이고, 깊이가 `maxDepth`에 닿으면 더 안 나눈다.
 * **다른 점 하나** — sklearn은 특성을 무작위 순서로 훑어 동점을 가르고, 여기는 **앞 열이
 * 이긴다.** 같은 데이터에서 언제나 같은 나무가 나오고, 동점이 아니면 sklearn과 같은 분할이다.
 */

import { REGRESSION_LEAF, type RegressionNode } from '../models/regression-tree'

export interface GrowInput {
  /** 열 우선 행렬. `columns[f][r]`가 r번 행의 f번 특성이다. */
  readonly columns: readonly Float64Array[]
  /** 행마다의 타깃(부스팅이면 잔차). */
  readonly targets: Float64Array
  /** 이 나무가 보는 행. **같은 행이 여러 번 와도 된다** — 배깅의 복원 추출이다. */
  readonly rows: Int32Array
  /** 이 깊이에 닿은 노드는 잎이다. 뿌리가 0이다. */
  readonly maxDepth: number
  /** 행이 이보다 적은 노드는 안 나눈다 (sklearn `min_samples_split`). */
  readonly minSplit: number
  /** 잎의 값. 없으면 타깃의 평균이다. */
  readonly leafValue?: (rows: Int32Array) => number
}

function mean(targets: Float64Array, rows: Int32Array): number {
  let sum = 0
  for (const row of rows) sum += targets[row] ?? 0
  return rows.length === 0 ? 0 : sum / rows.length
}

interface Split {
  readonly column: number
  readonly threshold: number
}

/**
 * 한 노드의 가장 좋은 분할. **없으면 `null`** — 행이 모자라거나, 타깃이 한 값이거나,
 * 어느 열로도 값이 안 갈리거나, 나눠도 오차가 안 줄 때다.
 */
function bestSplit(input: GrowInput, rows: Int32Array): Split | null {
  const { columns, targets } = input
  const n = rows.length
  let sum = 0
  let sumSquares = 0
  for (const row of rows) {
    const value = targets[row] ?? 0
    sum += value
    sumSquares += value * value
  }
  const parentScore = (sum * sum) / n
  const error = sumSquares - parentScore
  // 타깃이 한 값이면 나눌 것이 없다. 부동소수 먼지를 오차로 읽지 않게 상대 기준으로 본다.
  if (!(error > 1e-12 * Math.max(sumSquares, 1e-300))) return null

  let bestScore = parentScore
  let best: Split | null = null
  const order = Array.from(rows)

  columns.forEach((column, columnIndex) => {
    order.sort((a, b) => (column[a] ?? 0) - (column[b] ?? 0))
    let leftSum = 0
    for (let i = 0; i < n - 1; i += 1) {
      const row = order[i]!
      leftSum += targets[row] ?? 0
      const here = column[row] ?? 0
      const next = column[order[i + 1]!] ?? 0
      if (here === next) continue
      const leftCount = i + 1
      const rightSum = sum - leftSum
      const score = (leftSum * leftSum) / leftCount + (rightSum * rightSum) / (n - leftCount)
      // **엄격히 커야 바꾼다** — 동점이면 앞 열·앞 자리가 이긴다(머리말).
      if (score > bestScore) {
        bestScore = score
        let threshold = (here + next) / 2
        // 두 값이 이웃한 부동소수면 가운데가 `next`로 반올림된다. 그러면 `<=`가 `next`를
        // 왼쪽에 넣어 분할이 무너진다 — sklearn도 같은 자리에서 `here`로 물린다.
        if (threshold >= next) threshold = here
        best = { column: columnIndex, threshold }
      }
    }
  })

  // 줄어든 오차가 먼지 수준이면 나누지 않는다 — 나눠 봐야 같은 값을 두 잎에 적는다.
  if (best === null || bestScore - parentScore <= 1e-12 * error) return null
  return best
}

/**
 * 나무 하나를 키워 **전위 순서의 노드 배열**로 돌려준다. 자식은 언제나 자기보다 뒤에
 * 적힌다 — 해석기가 그 성질에 기대어 순환 없이 걷는다(`models/regression-tree.ts`).
 */
export function growRegressionTree(input: GrowInput): RegressionNode[] {
  const nodes: RegressionNode[] = []
  const leafOf = input.leafValue ?? ((rows: Int32Array) => mean(input.targets, rows))

  const grow = (rows: Int32Array, depth: number): number => {
    const index = nodes.length
    const split =
      depth < input.maxDepth && rows.length >= input.minSplit && rows.length >= 2
        ? bestSplit(input, rows)
        : null

    if (split === null) {
      nodes.push([REGRESSION_LEAF, leafOf(rows), REGRESSION_LEAF, REGRESSION_LEAF])
      return index
    }

    const column = input.columns[split.column]!
    const left: number[] = []
    const right: number[] = []
    for (const row of rows) {
      if ((column[row] ?? 0) <= split.threshold) left.push(row)
      else right.push(row)
    }

    // 자리를 먼저 잡고 자식을 적는다 — 자식 인덱스가 자기보다 크게 된다.
    nodes.push([REGRESSION_LEAF, 0, REGRESSION_LEAF, REGRESSION_LEAF])
    const leftIndex = grow(Int32Array.from(left), depth + 1)
    const rightIndex = grow(Int32Array.from(right), depth + 1)
    nodes[index] = [split.column, split.threshold, leftIndex, rightIndex]
    return index
  }

  grow(input.rows, 0)
  return nodes
}

/** 행 우선 행렬을 열 우선으로. 나무가 열 하나를 통째로 정렬하므로 이쪽이 빠르다. */
export function toColumns(features: readonly (readonly number[])[]): Float64Array[] {
  const width = features[0]?.length ?? 0
  const columns = Array.from({ length: width }, () => new Float64Array(features.length))
  features.forEach((row, r) => {
    for (let f = 0; f < width; f += 1) columns[f]![r] = row[f] ?? 0
  })
  return columns
}

/** 0부터 n−1까지의 행 번호. */
export function allRows(count: number): Int32Array {
  return Int32Array.from({ length: count }, (_, index) => index)
}
