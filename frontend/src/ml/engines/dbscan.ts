/**
 * DBSCAN — **밀도로 무리를 찾는다** (open-decisions.md "DBSCAN을 넣는다").
 *
 * K-평균은 무리 수를 사람이 정하고, 여기는 **반경(`eps`)과 최소 이웃 수(`min_samples`)**만
 * 정한다. 반경 안에 자기 포함 `min_samples`개 이상이 있는 점이 **핵심 점**이고, 핵심 점끼리
 * 반경으로 이어진 것이 한 무리다. 어느 핵심 점의 반경에도 안 드는 점은 **잡음(`-1`)**이다.
 *
 * **sklearn `DBSCAN(algorithm='brute')`과 같은 차례로 번호를 붙인다.** 행을 앞에서부터
 * 훑어 아직 번호가 없는 핵심 점을 만나면 새 무리를 열고, 그 무리를 이웃 차례(행 번호
 * 차례)로 넓힌다. 경계 점은 **먼저 닿은 무리**에 붙는다 — sklearn의 `dbscan_inner`와 같은
 * 규칙이라 같은 데이터에서 같은 번호가 나온다.
 *
 * **이웃 목록을 저장하지 않는다.** 반경이 크면 목록이 `행²`만큼 커져 교실 PC의 메모리를
 * 넘는다. 대신 거리를 두 번 잰다(핵심 점 판정 한 번, 무리를 넓힐 때 한 번) — 시간은
 * `O(행² × 특성)`으로 같고 메모리는 `O(행)`이다.
 *
 * **난수를 안 쓴다.** 같은 데이터·같은 손잡이면 언제나 같은 무리다.
 */

import { NOISE } from '../models/dbscan'

export interface DbscanResult {
  /** 행마다 무리 번호. 잡음은 `-1`이다. */
  readonly labels: Int32Array
  /** 행마다 핵심 점인가. */
  readonly core: Uint8Array
  /** 무리 수. 잡음은 세지 않는다. */
  readonly clusterCount: number
}

export function fitDbscan(
  features: readonly (readonly number[])[],
  eps: number,
  minSamples: number,
): DbscanResult {
  const n = features.length
  const width = features[0]?.length ?? 0
  const rows = features.map((row) => Float64Array.from(row))
  const reach = eps * eps

  const within = (a: Float64Array, b: Float64Array): boolean => {
    let distance = 0
    for (let column = 0; column < width; column += 1) {
      const gap = (a[column] ?? 0) - (b[column] ?? 0)
      distance += gap * gap
      if (distance > reach) return false
    }
    return true
  }

  // 1. 핵심 점 — 반경 안의 점을 **자기 자신까지** 센다(sklearn과 같다).
  const core = new Uint8Array(n)
  for (let i = 0; i < n; i += 1) {
    const row = rows[i]!
    let count = 0
    for (let j = 0; j < n && count < minSamples; j += 1) {
      if (within(row, rows[j]!)) count += 1
    }
    if (count >= minSamples) core[i] = 1
  }

  // 2. 무리 — 번호가 없는 핵심 점에서 열고, 핵심 점의 이웃으로만 넓힌다.
  const labels = new Int32Array(n).fill(NOISE)
  let clusterCount = 0
  const stack: number[] = []
  for (let start = 0; start < n; start += 1) {
    if (labels[start] !== NOISE || core[start] !== 1) continue
    let current = start
    for (;;) {
      if (labels[current] === NOISE) {
        labels[current] = clusterCount
        if (core[current] === 1) {
          const row = rows[current]!
          for (let j = 0; j < n; j += 1) {
            if (labels[j] === NOISE && within(row, rows[j]!)) stack.push(j)
          }
        }
      }
      const next = stack.pop()
      if (next === undefined) break
      current = next
    }
    clusterCount += 1
  }

  return { labels, core, clusterCount }
}

/**
 * 무리마다의 평균 — **화면과 지표가 쓰는 중심점이다.** DBSCAN 자체에는 중심점이 없다.
 * 무리가 둥글지 않을 수 있어 이 평균이 무리 밖에 떨어질 수도 있다 — 그것도 수업 장면이다.
 */
export function clusterMeans(
  features: readonly (readonly number[])[],
  labels: Int32Array,
  clusterCount: number,
): number[][] {
  const width = features[0]?.length ?? 0
  const sums = Array.from({ length: clusterCount }, () => new Array<number>(width).fill(0))
  const counts = new Array<number>(clusterCount).fill(0)
  features.forEach((row, index) => {
    const label = labels[index] ?? NOISE
    if (label < 0) return
    counts[label] = (counts[label] ?? 0) + 1
    const sum = sums[label]!
    for (let column = 0; column < width; column += 1)
      sum[column] = (sum[column] ?? 0) + (row[column] ?? 0)
  })
  return sums.map((sum, label) => sum.map((value) => value / Math.max(counts[label] ?? 1, 1)))
}
