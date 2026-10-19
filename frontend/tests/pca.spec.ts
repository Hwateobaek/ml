/**
 * 주성분 분석 (`ml/pca.ts`).
 *
 * **성질을 본다. 숫자를 못 박지 않는다.** 무작위 SVD라 끝자리는 씨앗을 따라가고,
 * sklearn과도 난수가 다르다(그 파일의 머리말). 대신 **주성분이 주성분이려면 반드시
 * 참이어야 하는 것** 넷을 본다 — 직교성, 분산이 큰 순서, 되살림 오차, 재현성.
 *
 * **셋째가 가장 중요하다.** 투영이 틀려도 모델은 예외 없이 끝까지 돌고 점수만 조금
 * 나빠진다. 되살림 오차가 그 조용한 실패를 잡는 유일한 자리다.
 */

import { describe, expect, it } from 'vitest'

import { applyProjection, fitProjection, type Projection } from '../src/ml/pca'

/**
 * **평면 위에 놓인 데이터.** 10차원이지만 실제로 자유로운 방향은 둘뿐이라,
 * 주성분 둘이면 원래 점을 그대로 되살릴 수 있어야 한다.
 */
function planeRows(count: number): number[][] {
  const first = [1, 0, 0.5, -0.2, 0, 0.3, 0, 0, 0.1, 0]
  const second = [0, 1, -0.3, 0.4, 0.2, 0, 0.1, 0, 0, 0.5]
  const rows: number[][] = []
  for (let index = 0; index < count; index += 1) {
    // 고르게 퍼진 두 계수. 난수를 안 쓰는 이유는 이 파일이 재현성도 보기 때문이다.
    const a = Math.sin(index * 0.7) * 3
    const b = Math.cos(index * 1.3) * 2
    rows.push(first.map((value, column) => value * a + (second[column] ?? 0) * b + 5))
  }
  return rows
}

/** 투영했다가 되돌린 점. 성분이 평면을 덮으면 원래 점과 같아야 한다. */
function restore(projection: Projection, row: readonly number[]): number[] {
  const coordinates = applyProjection(projection, row)
  return projection.mean.map((mean, column) =>
    coordinates.reduce(
      (sum, value, index) => sum + value * (projection.components[index]?.[column] ?? 0),
      mean,
    ),
  )
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0)
}

describe('주성분 분석', () => {
  it('평면 위의 데이터는 주성분 둘로 그대로 되살아난다', () => {
    const rows = planeRows(40)
    const projection = fitProjection(rows, 2, 42)

    expect(projection?.components).toHaveLength(2)
    for (const row of rows) {
      const back = restore(projection!, row)
      for (const [column, value] of row.entries()) {
        // 투영은 여섯 자리로 잘려 파일에 담긴다 (`ml/pca.ts`의 `rounded`).
        expect(back[column]).toBeCloseTo(value, 3)
      }
    }
  })

  it('성분들이 정규직교다', () => {
    const projection = fitProjection(planeRows(40), 2, 7)
    const [first, second] = projection?.components ?? []

    expect(dot(first ?? [], first ?? [])).toBeCloseTo(1, 4)
    expect(dot(second ?? [], second ?? [])).toBeCloseTo(1, 4)
    // 직교하지 않으면 같은 방향을 두 번 세는 좌표가 된다.
    expect(dot(first ?? [], second ?? [])).toBeCloseTo(0, 4)
  })

  it('분산이 큰 성분이 앞에 선다', () => {
    const rows = planeRows(60)
    const projection = fitProjection(rows, 2, 11)
    const coordinates = rows.map((row) => applyProjection(projection!, row))

    const spread = (index: number): number => {
      const values = coordinates.map((coordinate) => coordinate[index] ?? 0)
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length
      return values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
    }

    expect(spread(0)).toBeGreaterThan(spread(1))
  })

  it('같은 씨앗이면 같은 성분이다', () => {
    const rows = planeRows(30)
    expect(fitProjection(rows, 2, 5)).toEqual(fitProjection(rows, 2, 5))
  })

  /**
   * **부호는 수학적으로 뜻이 없지만 화면에는 뜻이 있다** (`ml/pca.ts`의 `flipSigns`).
   * 규칙이 없으면 같은 데이터가 씨앗마다 뒤집힌 좌표를 주고, 학생은 그림이 뒤집힌
   * 이유를 못 찾는다. sklearn의 `svd_flip`과 같은 규칙을 쓴다.
   */
  it('성분마다 절댓값이 가장 큰 칸이 양수다', () => {
    const projection = fitProjection(planeRows(30), 2, 99)

    for (const component of projection?.components ?? []) {
      const largest = component.reduce(
        (best, value) => (Math.abs(value) > Math.abs(best) ? value : best),
        0,
      )
      expect(largest).toBeGreaterThan(0)
    }
  })

  it('씨앗이 달라도 같은 평면을 찾는다', () => {
    const rows = planeRows(40)
    const first = fitProjection(rows, 2, 1)
    const second = fitProjection(rows, 2, 2)

    // 두 벌의 성분이 같은 평면을 덮으면 되살린 점이 같다 — 축의 순서나 부호가 아니라
    // **무엇을 설명하는가**가 같아야 한다.
    for (const row of rows.slice(0, 5)) {
      const left = restore(first!, row)
      const right = restore(second!, row)
      for (const [column, value] of left.entries()) {
        expect(right[column]).toBeCloseTo(value, 3)
      }
    }
  })

  it('줄일 것이 없으면 안 배운다', () => {
    const rows = planeRows(20)
    // 열 수만큼(이상) 남기는 것은 줄이는 것이 아니다.
    expect(fitProjection(rows, 10, 3)).toBeNull()
    expect(fitProjection(rows, 11, 3)).toBeNull()
    // 행이 없거나 열이 없는 표.
    expect(fitProjection([], 2, 3)).toBeNull()
    expect(fitProjection([[], []], 2, 3)).toBeNull()
  })

  /**
   * **행보다 많은 성분을 요구할 수 있다.** 사진 20장에서 50개를 달라고 하면 20개까지만
   * 뜻이 있고, 그 뒤는 분산이 0인 방향이다 — 0으로 나누지 않고 거기서 멈춘다.
   */
  it('행보다 많이 달라고 하면 가진 만큼만 준다', () => {
    const projection = fitProjection(planeRows(6), 8, 4)

    expect(projection).not.toBeNull()
    expect((projection?.components.length ?? 0) <= 6).toBe(true)
    // 평면 위의 데이터라 뜻이 있는 방향은 둘뿐이다.
    expect(projection?.components.length).toBeGreaterThanOrEqual(2)
  })

  it('예측 한 줄도 학습과 같은 좌표를 받는다', () => {
    const rows = planeRows(30)
    const projection = fitProjection(rows, 2, 8)
    const row = rows[3] as number[]

    // 같은 행을 따로 넣어도 학습 때와 같은 자리에 놓여야 한다 — 두 벌이 되면 예측만
    // 다른 좌표계에 선다.
    expect(applyProjection(projection!, [...row])).toEqual(applyProjection(projection!, row))
  })
})
