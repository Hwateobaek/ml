/**
 * 이상치의 경계와 개수 (open-decisions.md "이상치는 훈련 데이터의 IQR로 클리핑한다").
 *
 * **판정 규칙이 여기 한 벌뿐이다.** 시각화의 상자그림, 데이터 요약의 이상치 수, 전처리기의
 * 클리핑 경계가 전부 이 함수들을 부른다 — 둘이 되면 "그림에서는 이상치인데 안 잘렸다"가
 * 생기고, 학생은 어느 쪽이 맞는지 알 수 없다.
 *
 * **무엇으로 부르는지는 부르는 쪽이 정한다.** 화면은 전체 데이터를, `fitPreprocessor`는
 * 훈련 데이터를 넘긴다. 이 파일은 받은 값만 본다.
 *
 * 검사는 `tests/outliers.spec.ts`가 붙는다.
 */

import { OUTLIER_IQR_MULTIPLIER } from '../limits'

export interface Quartiles {
  readonly q1: number
  readonly median: number
  readonly q3: number
}

/** 이 경계 밖이 이상치다. 경계값 자체는 이상치가 아니다. */
export interface OutlierBounds {
  readonly low: number
  readonly high: number
}

/** 정렬된 값에서 분위수. 사이에 있으면 선형으로 잇는다 (numpy `percentile`의 기본과 같다). */
function quantile(sorted: readonly number[], fraction: number): number {
  const position = (sorted.length - 1) * fraction
  const low = Math.floor(position)
  const high = Math.ceil(position)
  const lowValue = sorted[low] ?? 0
  if (low === high) return lowValue
  return lowValue + ((sorted[high] ?? 0) - lowValue) * (position - low)
}

/** **정렬된** 값의 사분위수. 값이 없으면 `null`이다. */
export function quartilesOfSorted(sorted: readonly number[]): Quartiles | null {
  if (sorted.length === 0) return null
  return {
    q1: quantile(sorted, 0.25),
    median: quantile(sorted, 0.5),
    q3: quantile(sorted, 0.75),
  }
}

/**
 * 이상치의 경계. **자를 수 없으면 `null`이다** — 값이 없거나 IQR이 0일 때다.
 *
 * **IQR이 0이면 경계를 안 준다.** 값 대부분이 같은 열에서는 Q1과 Q3가 같아 나머지 값이
 * 전부 이상치가 되고, 그 경계로 자르면 **열이 한 값이 된다.** 0과 1만 있는 수치 열이
 * 흔해서 드문 경우가 아니다.
 */
export function outlierBounds(values: readonly number[]): OutlierBounds | null {
  const sorted = [...values].sort((left, right) => left - right)
  const quartiles = quartilesOfSorted(sorted)
  if (quartiles === null) return null
  const spread = quartiles.q3 - quartiles.q1
  if (!(spread > 0)) return null
  return {
    low: quartiles.q1 - OUTLIER_IQR_MULTIPLIER * spread,
    high: quartiles.q3 + OUTLIER_IQR_MULTIPLIER * spread,
  }
}

/** 경계 밖인가. 경계가 없으면 아무것도 이상치가 아니다. */
export function isOutlier(value: number, bounds: OutlierBounds | null): boolean {
  return bounds !== null && (value < bounds.low || value > bounds.high)
}

/** 경계 안으로 밀어 넣는다. 경계가 없으면 그대로다. */
export function clipValue(value: number, bounds: OutlierBounds | null): number {
  if (bounds === null) return value
  if (value < bounds.low) return bounds.low
  if (value > bounds.high) return bounds.high
  return value
}

/** 경계 밖 값의 개수. **경계를 같은 값에서 구한다** — 화면이 전체 데이터로 셀 때 쓴다. */
export function countOutliers(values: readonly number[]): number {
  const bounds = outlierBounds(values)
  if (bounds === null) return 0
  let count = 0
  for (const value of values) if (isOutlier(value, bounds)) count += 1
  return count
}
