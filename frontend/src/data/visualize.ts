/**
 * 시각화 단계가 그리는 것들의 **계산**. 그림은 여기 없다.
 *
 * **화면 밖에 있는 이유는 이 저장소의 다른 계산과 같다** (`CLAUDE.md` §4의 "검증 가능한
 * 로직을 컴포넌트 밖으로 빼라"). 구간을 나누고 상관을 재는 것은 눈으로 봐서는 틀린 줄
 * 모르는 종류라, 화면을 마운트해야만 부를 수 있으면 아무도 검사하지 않는다.
 * 검사는 `tests/visualize.spec.ts`가 붙는다.
 *
 * **여기는 Chart.js를 모른다.** 이 파일이 내놓는 것은 숫자이고, 그것을 데이터셋 모양으로
 * 바꾸는 것은 화면이다 — `ml/cluster-chart.ts`가 차트 설정을 갖는 것과 갈라지는 자리다.
 */

import { HISTOGRAM_BIN_COUNT, OUTLIER_MARK_COUNT } from '@/limits'
import {
  countOutliers,
  isOutlier,
  outlierBounds,
  quartilesOfSorted,
  type OutlierBounds,
} from '@/ml/outliers'
import type { Dataset } from '@/ml/preprocess'

/** 히스토그램 한 칸. 경계는 `[start, end)`이고 마지막 칸만 끝을 포함한다. */
export interface HistogramBin {
  readonly start: number
  readonly end: number
  readonly count: number
}

/**
 * 열 하나의 수치값. **빈 칸과 숫자가 아닌 값은 빠진다.**
 *
 * 빠뜨리지 않고 0으로 채우면 분포가 0에 산을 하나 만들고, 그것이 결측인지 진짜 0인지
 * 화면에서 갈라지지 않는다. 결측 수는 이미 `data/columns.ts`의 열 요약이 말한다.
 */
export function numericValues(dataset: Dataset, column: string): number[] {
  const index = dataset.columns.indexOf(column)
  if (index < 0) return []
  const values: number[] = []
  for (const row of dataset.rows) {
    const cell = row[index]
    if (cell === undefined || cell.trim() === '') continue
    const value = Number(cell)
    if (Number.isFinite(value)) values.push(value)
  }
  return values
}

/**
 * 두 열을 **같은 행에서** 함께 읽는다. 한쪽이라도 비면 그 행을 버린다.
 *
 * **행을 맞추는 것이 요점이다.** 열마다 따로 `numericValues`를 불러 짝지으면 결측이
 * 있는 열에서 길이가 어긋나고, 그러면 산점도의 점이 남의 행과 묶인다.
 */
export function numericPairs(
  dataset: Dataset,
  xColumn: string,
  yColumn: string,
): { x: number; y: number; row: number }[] {
  const xIndex = dataset.columns.indexOf(xColumn)
  const yIndex = dataset.columns.indexOf(yColumn)
  if (xIndex < 0 || yIndex < 0) return []
  const pairs: { x: number; y: number; row: number }[] = []
  dataset.rows.forEach((row, position) => {
    const x = Number(row[xIndex])
    const y = Number(row[yIndex])
    const xEmpty = (row[xIndex] ?? '').trim() === ''
    const yEmpty = (row[yIndex] ?? '').trim() === ''
    if (xEmpty || yEmpty || !Number.isFinite(x) || !Number.isFinite(y)) return
    pairs.push({ x, y, row: position })
  })
  return pairs
}

/**
 * 값들을 같은 너비의 구간으로 나눈다. 값이 없으면 빈 배열이다.
 *
 * **값이 전부 같으면 구간을 하나만 만든다.** 너비가 0이면 나눗셈이 무한대가 되고
 * Chart.js에 NaN이 흘러 들어가 축이 통째로 사라진다.
 */
export function histogram(values: readonly number[], bins = HISTOGRAM_BIN_COUNT): HistogramBin[] {
  if (values.length === 0) return []
  // **`Math.min(...values)`를 쓰지 않는다** (`tests/spread-rules.spec.ts`). 행 수만큼을
  // 인자로 펼치면 큰 표에서 스택이 넘친다 — 여기 오는 배열의 길이가 곧 행 수다.
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const value of values) {
    if (value < min) min = value
    if (value > max) max = value
  }
  if (min === max) return [{ start: min, end: max, count: values.length }]

  const width = (max - min) / bins
  const counts = new Array<number>(bins).fill(0)
  for (const value of values) {
    // 마지막 칸은 끝을 포함한다 — 안 그러면 최댓값 하나가 어느 칸에도 안 들어간다.
    const slot = Math.min(bins - 1, Math.floor((value - min) / width))
    counts[slot] = (counts[slot] ?? 0) + 1
  }
  return counts.map((count, index) => ({
    start: min + index * width,
    end: min + (index + 1) * width,
    count,
  }))
}

/** 다섯 수 요약. 상자그림이 이것으로 선다. */
export interface FiveNumbers {
  readonly min: number
  readonly q1: number
  readonly median: number
  readonly q3: number
  readonly max: number
}

/**
 * 값이 없으면 null이다. 화면은 그때 상자를 안 그린다.
 *
 * **사분위수는 `ml/outliers.ts`의 것을 쓴다.** 여기 따로 두면 상자의 Q1·Q3과 이상치
 * 경계를 만든 Q1·Q3이 다른 규칙에서 나올 수 있고, 그러면 그림의 점과 전처리가 자른 값이
 * 어긋난다.
 */
export function fiveNumbers(values: readonly number[]): FiveNumbers | null {
  const sorted = [...values].sort((left, right) => left - right)
  const quartiles = quartilesOfSorted(sorted)
  if (quartiles === null) return null
  return {
    min: sorted[0] ?? 0,
    q1: quartiles.q1,
    median: quartiles.median,
    q3: quartiles.q3,
    max: sorted[sorted.length - 1] ?? 0,
  }
}

/** 데이터 통계 표의 한 열. pandas `describe()`가 내는 여덟 칸과 같은 순서다. */
export interface DescriptiveStats extends FiveNumbers {
  readonly count: number
  readonly mean: number
  /** 표본 표준편차(n−1로 나눈다). **값이 하나뿐이면 없다** — 나눌 수가 0이다. */
  readonly std: number | null
}

/**
 * 개수·평균·표준편차·다섯 수 요약. 값이 없으면 null이다.
 *
 * **표준편차는 n−1로 나눈다.** 교과서의 통계표가 pandas `describe()`로 뽑은 것이고 그쪽이
 * 표본 표준편차다 — n으로 나누면 같은 CSV에서 교과서와 다른 숫자가 나온다.
 *
 * 사분위수는 `fiveNumbers`의 것이라 상자그림과 같은 규칙이다(numpy 선형 보간 = pandas 기본).
 */
export function descriptiveStats(values: readonly number[]): DescriptiveStats | null {
  const summary = fiveNumbers(values)
  if (summary === null) return null
  const count = values.length
  let sum = 0
  for (const value of values) sum += value
  const mean = sum / count
  let squares = 0
  for (const value of values) squares += (value - mean) ** 2
  const std = count > 1 ? Math.sqrt(squares / (count - 1)) : null
  return { count, mean, std, ...summary }
}

/** 상자그림 한 개. 좌표가 아니라 **값**이다 — 어디에 그릴지는 화면이 정한다. */
export interface BoxPlot extends FiveNumbers {
  /** 이상치 경계. IQR이 0이면 없다 (`outlierBounds`). */
  readonly bounds: OutlierBounds | null
  /** 수염의 끝. **경계 안에서 가장 먼 실제 값이다** — 경계값 자체가 아니다. */
  readonly whiskerLow: number
  readonly whiskerHigh: number
  /** 이상치의 개수. 아래 `marks`가 잘렸어도 이것은 전부 센다. */
  readonly outlierCount: number
  /** 점으로 찍을 이상치. **서로 다른 값만, `OUTLIER_MARK_COUNT`개까지** 준다. */
  readonly marks: readonly number[]
}

/**
 * 상자그림의 값. 값이 없으면 null이다.
 *
 * **점은 서로 다른 값만 찍는다.** 같은 값이 백 번 나와도 같은 자리에 겹치는 점이라 더
 * 보이는 것이 없고, 개수는 `outlierCount`가 말한다.
 */
export function boxPlot(values: readonly number[]): BoxPlot | null {
  const summary = fiveNumbers(values)
  if (summary === null) return null
  const bounds = outlierBounds(values)

  let whiskerLow = Number.POSITIVE_INFINITY
  let whiskerHigh = Number.NEGATIVE_INFINITY
  let outlierCount = 0
  const marks = new Set<number>()
  for (const value of values) {
    if (isOutlier(value, bounds)) {
      outlierCount += 1
      if (marks.size < OUTLIER_MARK_COUNT) marks.add(value)
      continue
    }
    if (value < whiskerLow) whiskerLow = value
    if (value > whiskerHigh) whiskerHigh = value
  }

  return {
    ...summary,
    bounds,
    whiskerLow,
    whiskerHigh,
    outlierCount,
    marks: [...marks].sort((left, right) => left - right),
  }
}

/** 범위를 정할 때 옆에 보여줄 참고값. */
export interface RangeGuide {
  readonly min: number
  readonly max: number
  /** IQR 경계. 사분위 범위가 0이면 없다. */
  readonly bounds: OutlierBounds | null
}

/**
 * 범위 입력칸 옆의 참고값 — **전체 데이터**의 최솟값·최댓값과 IQR 경계. 숫자가 없으면 null이다.
 *
 * **상자그림과 같은 함수에서 나온다** (`boxPlot`). 시각화 화면에서 본 경계와 여기 적힌
 * 경계가 같아야 학생이 그림을 보고 숫자를 옮겨 적을 수 있다.
 */
export function rangeGuide(dataset: Dataset, column: string): RangeGuide | null {
  const plot = boxPlot(numericValues(dataset, column))
  return plot === null ? null : { min: plot.min, max: plot.max, bounds: plot.bounds }
}

/**
 * 열마다 이상치 개수 (`ml/outliers.ts`의 `countOutliers`). **받은 표 전체로 센다** —
 * 데이터 요약과 열 표가 쓰고, 둘 다 "전체 데이터"를 보여주는 자리다.
 *
 * **숫자로 된 값이 하나도 없는 열은 목록에 없다.** 0과 "셀 수 없음"은 다른 말이라,
 * 화면이 없는 열에는 수 대신 빈 표시를 찍는다.
 */
export function outlierCounts(dataset: Dataset, names: readonly string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const name of names) {
    const values = numericValues(dataset, name)
    if (values.length > 0) counts.set(name, countOutliers(values))
  }
  return counts
}

/**
 * 피어슨 상관계수. 값이 둘 미만이거나 한쪽이 상수면 0이다.
 *
 * **상수 열에서 0을 내놓는 것은 선택이다.** 분모가 0이라 수학적으로는 정의되지 않는데,
 * NaN을 흘리면 히트맵의 그 칸이 색 없이 비어 학생이 "고장"으로 읽는다. 0은 "같이 움직인다는
 * 증거가 없다"이고, 상수 열에 대해 그것은 참이다.
 */
export function correlation(left: readonly number[], right: readonly number[]): number {
  const length = Math.min(left.length, right.length)
  if (length < 2) return 0
  let sumLeft = 0
  let sumRight = 0
  for (let index = 0; index < length; index += 1) {
    sumLeft += left[index] ?? 0
    sumRight += right[index] ?? 0
  }
  const meanLeft = sumLeft / length
  const meanRight = sumRight / length

  let product = 0
  let squaresLeft = 0
  let squaresRight = 0
  for (let index = 0; index < length; index += 1) {
    const deltaLeft = (left[index] ?? 0) - meanLeft
    const deltaRight = (right[index] ?? 0) - meanRight
    product += deltaLeft * deltaRight
    squaresLeft += deltaLeft * deltaLeft
    squaresRight += deltaRight * deltaRight
  }
  const spread = Math.sqrt(squaresLeft * squaresRight)
  return spread === 0 ? 0 : product / spread
}

/**
 * 열들끼리의 상관 행렬. `matrix[i][j]`가 `columns[i]`와 `columns[j]`의 상관이다.
 *
 * **짝마다 행을 다시 맞춘다** (`numericPairs`). 열 전체를 한 번만 읽어 두고 쓰면 결측이
 * 있는 열에서 길이가 어긋난 채로 곱해진다.
 */
export function correlationMatrix(dataset: Dataset, columns: readonly string[]): number[][] {
  return columns.map((rowColumn) =>
    columns.map((cellColumn) => {
      if (rowColumn === cellColumn) return 1
      const pairs = numericPairs(dataset, rowColumn, cellColumn)
      return correlation(
        pairs.map((pair) => pair.x),
        pairs.map((pair) => pair.y),
      )
    }),
  )
}

/* ── 범주 ── */

/**
 * 범주 값의 차례. **숫자가 섞인 이름은 수의 크기로 놓는다** — `1등급, 10등급, 2등급`이
 * 아니라 `1, 2, 10`이다. 로케일을 고정하는 것은 같은 파일이 컴퓨터마다 다른 순서로
 * 그려지지 않게 하려는 것이다.
 */
const categoryOrder = new Intl.Collator('ko', { numeric: true })

function sortCategories(values: Iterable<string>): string[] {
  return [...values].sort((left, right) => categoryOrder.compare(left, right))
}

/** 셀의 범주 값. **빈 칸은 범주가 아니다** — `null`로 돌려 세는 쪽이 빼게 한다. */
function categoryOf(row: readonly string[], index: number): string | null {
  const cell = (row[index] ?? '').trim()
  return cell === '' ? null : cell
}

/** 범주 하나의 개수. */
export interface CategoryCount {
  readonly value: string
  readonly count: number
}

/**
 * 열의 값별 개수 — 막대그래프가 이것으로 선다. 값은 이름 차례이고, **빈 칸은 세지 않는다.**
 *
 * **개수 차례로 놓지 않는다.** `0/1`, `1등급/2등급/3등급`처럼 순서가 뜻인 열이 흔해서,
 * 많은 순으로 섞으면 학생이 축을 읽다가 헷갈린다. 많고 적음은 막대 높이가 말한다.
 */
export function categoryCounts(dataset: Dataset, column: string): CategoryCount[] {
  const index = dataset.columns.indexOf(column)
  if (index < 0) return []
  const counts = new Map<string, number>()
  for (const row of dataset.rows) {
    const value = categoryOf(row, index)
    if (value !== null) counts.set(value, (counts.get(value) ?? 0) + 1)
  }
  return sortCategories(counts.keys()).map((value) => ({ value, count: counts.get(value) ?? 0 }))
}

/** 범주 하나의 상자그림. */
export interface GroupBoxPlot {
  readonly group: string
  readonly plot: BoxPlot
}

/**
 * 수치 열 하나를 범주별로 나눈 상자그림들. 범주는 이름 차례다.
 *
 * **범주가 비었거나 수가 아닌 행은 빠진다.** 숫자가 하나도 없는 범주는 목록에 없다 —
 * 빈 상자를 그리면 "그 범주는 값이 0 근처"로 읽힌다.
 *
 * **이상치 경계는 범주마다 따로다.** 전처리가 자르는 경계(열 전체)와 다를 수 있고, 그것이
 * 요점이다 — 한 종 안에서 튀는 값과 열 전체에서 튀는 값은 다른 질문이다.
 */
export function groupedBoxPlots(
  dataset: Dataset,
  valueColumn: string,
  groupColumn: string,
): GroupBoxPlot[] {
  const valueIndex = dataset.columns.indexOf(valueColumn)
  const groupIndex = dataset.columns.indexOf(groupColumn)
  if (valueIndex < 0 || groupIndex < 0) return []
  const buckets = new Map<string, number[]>()
  for (const row of dataset.rows) {
    const group = categoryOf(row, groupIndex)
    const cell = (row[valueIndex] ?? '').trim()
    const value = Number(cell)
    if (group === null || cell === '' || !Number.isFinite(value)) continue
    const bucket = buckets.get(group)
    if (bucket === undefined) buckets.set(group, [value])
    else bucket.push(value)
  }
  const plots: GroupBoxPlot[] = []
  for (const group of sortCategories(buckets.keys())) {
    const plot = boxPlot(buckets.get(group) ?? [])
    if (plot !== null) plots.push({ group, plot })
  }
  return plots
}

/** 교차표. `counts[i][j]`가 `rows[i]`이면서 `columns[j]`인 행의 수다. */
export interface CrossTab {
  readonly rows: readonly string[]
  readonly columns: readonly string[]
  readonly counts: readonly (readonly number[])[]
}

/**
 * 두 범주 열의 조합별 개수. **둘 중 하나라도 빈 칸인 행은 세지 않는다.**
 *
 * 한 번도 안 나온 조합도 칸이 있고 값이 0이다 — "그 섬에는 그 종이 없다"가 이 표에서
 * 가장 많이 읽히는 사실이라, 칸을 빼면 그것이 안 보인다.
 */
export function crossTab(dataset: Dataset, rowColumn: string, columnColumn: string): CrossTab {
  const rowIndex = dataset.columns.indexOf(rowColumn)
  const columnIndex = dataset.columns.indexOf(columnColumn)
  if (rowIndex < 0 || columnIndex < 0) return { rows: [], columns: [], counts: [] }
  const pairs = new Map<string, Map<string, number>>()
  const columnValues = new Set<string>()
  for (const row of dataset.rows) {
    const left = categoryOf(row, rowIndex)
    const right = categoryOf(row, columnIndex)
    if (left === null || right === null) continue
    columnValues.add(right)
    const line = pairs.get(left) ?? new Map<string, number>()
    line.set(right, (line.get(right) ?? 0) + 1)
    pairs.set(left, line)
  }
  const rows = sortCategories(pairs.keys())
  const columns = sortCategories(columnValues)
  return {
    rows,
    columns,
    counts: rows.map((left) => columns.map((right) => pairs.get(left)?.get(right) ?? 0)),
  }
}

/* ── 선그래프 ── */

/** 선 하나. `values[i]`가 `labels[i]` 자리의 값이고, 값이 없으면 `null`이라 선이 끊긴다. */
export interface LineSeries {
  readonly column: string
  readonly values: readonly (number | null)[]
}

export interface LineChart {
  readonly labels: readonly string[]
  readonly series: readonly LineSeries[]
  /** 같은 가로축 값을 가진 행이 둘 이상이라 평균을 냈는가. 화면이 그 사실을 말한다. */
  readonly averaged: boolean
}

/**
 * 가로축 열의 값마다 세로축 열들의 값을 늘어놓는다 — 연도별 물가 같은 선그래프다.
 *
 * **가로축 값이 같은 행이 여럿이면 평균을 낸다.** 한 해에 여러 번 잰 표에서 점을 다 찍으면
 * 선이 한 자리에서 위아래로 긁히고, 추세가 안 보인다. 평균을 냈는지는 `averaged`가 말한다.
 *
 * **가로축은 차례가 있는 값으로 놓는다.** 값이 전부 숫자면 수의 크기로, 아니면 숫자가 섞인
 * 이름 차례로 놓는다(`2002`, `2010` / `1월`, `10월`). 파일에 적힌 차례를 따르지 않는 것은
 * 최신 연도가 위에 오는 표가 흔하기 때문이다 — 그대로 그리면 선이 거꾸로 간다.
 *
 * 가로축이 빈 칸인 행은 빠진다. 세로값이 비었거나 숫자가 아니면 그 칸만 빠진다.
 */
export function lineChart(
  dataset: Dataset,
  xColumn: string,
  yColumns: readonly string[],
): LineChart {
  const xIndex = dataset.columns.indexOf(xColumn)
  const yIndexes = yColumns.map((column) => dataset.columns.indexOf(column))
  if (xIndex < 0) return { labels: [], series: [], averaged: false }

  const groups = new Map<string, { count: number; sums: number[]; counts: number[] }>()
  for (const row of dataset.rows) {
    const x = categoryOf(row, xIndex)
    if (x === null) continue
    let group = groups.get(x)
    if (group === undefined) {
      group = { count: 0, sums: yIndexes.map(() => 0), counts: yIndexes.map(() => 0) }
      groups.set(x, group)
    }
    group.count += 1
    yIndexes.forEach((index, position) => {
      if (index < 0) return
      const cell = (row[index] ?? '').trim()
      const value = Number(cell)
      if (cell === '' || !Number.isFinite(value)) return
      group.sums[position] = (group.sums[position] ?? 0) + value
      group.counts[position] = (group.counts[position] ?? 0) + 1
    })
  }

  const keys = [...groups.keys()]
  const allNumeric = keys.every((key) => Number.isFinite(Number(key)))
  const labels = allNumeric
    ? keys.sort((left, right) => Number(left) - Number(right))
    : sortCategories(keys)

  let averaged = false
  for (const group of groups.values()) if (group.count > 1) averaged = true

  return {
    labels,
    series: yColumns.map((column, position) => ({
      column,
      values: labels.map((label) => {
        const group = groups.get(label)
        const count = group?.counts[position] ?? 0
        return count === 0 ? null : (group?.sums[position] ?? 0) / count
      }),
    })),
    averaged,
  }
}

/** 이름이 시간을 가리키는 열. 학생의 표는 한글 머리글이 흔해서 두 언어를 함께 본다. */
const TIME_NAME = /year|date|month|day|time|연도|년|날짜|월|일자|시간|시기/i

/**
 * 선그래프 가로축의 첫 값. **이름이 시간을 가리키는 열을 먼저 잡는다** — 선그래프는 거의
 * 늘 시간의 흐름을 그리고, 표의 첫 열은 흔히 번호나 이름이다. 그런 열이 없으면 첫 열이다.
 */
export function defaultLineAxis(columns: readonly string[]): string {
  return columns.find((name) => TIME_NAME.test(name)) ?? columns[0] ?? ''
}
