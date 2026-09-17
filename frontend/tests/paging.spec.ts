/**
 * `data/paging.ts` — 쪽 나눔의 경계값.
 *
 * **마지막 쪽이 이 검사의 요점이다.** 딱 떨어지는 경우와 한 행만 남는 경우가 갈리고,
 * 틀리면 학생이 자기 데이터의 끝을 못 본다.
 */

import { describe, expect, it } from 'vitest'

import { clampPage, pageCount, pageItems, pageRange } from '@/data/paging'

describe('pageCount', () => {
  it('딱 떨어지면 나눈 만큼이다', () => {
    expect(pageCount(200, 100)).toBe(2)
  })

  it('남으면 한 쪽 더다', () => {
    expect(pageCount(201, 100)).toBe(3)
  })

  it('행이 없어도 한 쪽이다', () => {
    expect(pageCount(0, 100)).toBe(1)
  })

  it('크기가 0이어도 죽지 않는다', () => {
    expect(pageCount(50, 0)).toBe(1)
  })
})

describe('clampPage', () => {
  it('범위를 넘으면 마지막 쪽이다', () => {
    expect(clampPage(99, 201, 100)).toBe(3)
  })

  it('0이나 음수는 첫 쪽이다', () => {
    expect(clampPage(0, 201, 100)).toBe(1)
    expect(clampPage(-5, 201, 100)).toBe(1)
  })

  it('수가 아니면 첫 쪽이다', () => {
    expect(clampPage(Number.NaN, 201, 100)).toBe(1)
  })
})

describe('pageRange', () => {
  it('첫 쪽은 0에서 시작한다', () => {
    expect(pageRange(1, 201, 100)).toEqual({ start: 0, end: 100 })
  })

  it('마지막 쪽은 남은 만큼만 덮는다', () => {
    expect(pageRange(3, 201, 100)).toEqual({ start: 200, end: 201 })
  })

  it('행이 없으면 빈 구간이다', () => {
    expect(pageRange(1, 0, 100)).toEqual({ start: 0, end: 0 })
  })
})

describe('pageItems', () => {
  const items = Array.from({ length: 201 }, (_, index) => index)

  it('쪽을 이어 붙이면 원래 목록이 된다 - 빠지거나 겹치는 행이 없다', () => {
    const joined = [
      ...pageItems(items, 1, 100),
      ...pageItems(items, 2, 100),
      ...pageItems(items, 3, 100),
    ]
    expect(joined).toEqual(items)
  })

  it('마지막 쪽에 한 행만 남는 경우를 잘라 낸다', () => {
    expect(pageItems(items, 3, 100)).toEqual([200])
  })

  it('원본을 안 건드린다', () => {
    const before = [...items]
    pageItems(items, 2, 100)
    expect(items).toEqual(before)
  })
})
