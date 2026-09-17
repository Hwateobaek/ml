/**
 * 사람이 정한 범위 — 이상치 처리의 `range` (open-decisions.md "이상치는 훈련 데이터의
 * IQR로 클리핑한다"의 "사람이 정한 범위로는 행을 뺀다").
 *
 * **여기서 지키는 것은 "같은 규칙이 모든 자리에 걸린다"이다.** 범위는 행을 빼는 일이라,
 * 훈련 데이터·따로 받은 테스트 데이터·화면의 행 수 가운데 하나라도 범위를 모르면 화면이
 * 말하는 숫자와 학습이 쓰는 행이 갈린다.
 */

import { describe, expect, it } from 'vitest'

import { boxPlot, rangeGuide } from '../src/data/visualize'
import { planRun } from '../src/ml/plan'
import { usableRows, type Dataset } from '../src/ml/preprocess'
import {
  activeRanges,
  appliedRanges,
  describeRange,
  hasBound,
  inRange,
  parseBound,
  rangeProblem,
  withColumnRange,
  type ColumnRanges,
} from '../src/ml/ranges'
import { rowUsage, trainableRowCount } from '../src/ml/selection'
import type { Settings, TabularSettings } from '../src/project/schema'

describe('범위 하나', () => {
  it('경계값은 범위 안이다', () => {
    expect(inRange(10, { min: 10, max: 20 })).toBe(true)
    expect(inRange(20, { min: 10, max: 20 })).toBe(true)
    expect(inRange(9.99, { min: 10, max: 20 })).toBe(false)
  })

  it('한쪽만 적어도 된다', () => {
    expect(inRange(1000, { min: 10 })).toBe(true)
    expect(inRange(-1000, { max: 20 })).toBe(true)
  })

  it('둘 다 비면 범위가 아니다', () => {
    expect(hasBound({})).toBe(false)
    expect(hasBound({ min: 0 })).toBe(true)
  })

  it('뒤집힌 범위를 이유로 말한다', () => {
    expect(rangeProblem({ min: 20, max: 10 })).toBe('reversed')
    expect(rangeProblem({ min: 10, max: 10 })).toBeNull()
    expect(rangeProblem({ min: 10 })).toBeNull()
  })
})

describe('대응표', () => {
  it('range를 고른 동안만 행을 뺀다 - 다른 방식이면 남은 숫자가 안 쓰인다', () => {
    const ranges: ColumnRanges = { x: { min: 1 } }
    expect(activeRanges({ outliers: 'range', ranges })).toBe(ranges)
    expect(activeRanges({ outliers: 'clip', ranges })).toBeUndefined()
    expect(activeRanges({ ranges })).toBeUndefined()
  })

  it('둘 다 비우면 항목을 지운다', () => {
    expect(withColumnRange({ x: { min: 1 } }, 'x', {})).toEqual({})
  })

  it('넘겨받은 대응표를 안 건드린다', () => {
    const before: ColumnRanges = { x: { min: 1 } }
    withColumnRange(before, 'y', { max: 2 })
    expect(before).toEqual({ x: { min: 1 } })
  })

  it('안 적은 쪽은 키로도 안 남긴다', () => {
    expect(withColumnRange(undefined, 'x', { min: 1, max: undefined })).toEqual({ x: { min: 1 } })
  })

  it('지금 쓰는 열의 범위만, 이름 순서로 준다', () => {
    const ranges: ColumnRanges = { b: { min: 1 }, unused: { max: 3 }, a: { max: 2 }, t: { min: 0 } }
    expect(appliedRanges(ranges, ['b', 'a'], 't').map(([name]) => name)).toEqual(['a', 'b', 't'])
  })

  it('이력에 적을 줄은 구간 기호다', () => {
    expect(describeRange(['키', { min: 150, max: 190 }])).toBe('키 [150, 190]')
    expect(describeRange(['키', { min: 150 }])).toBe('키 [150, ∞]')
    expect(describeRange(['키', { max: 190 }])).toBe('키 [−∞, 190]')
  })
})

const table: Dataset = {
  columns: ['키', '몸무게', '반'],
  rows: [
    ['150', '40', 'A'],
    ['170', '60', 'B'],
    ['300', '65', 'A'], // 키가 범위 밖
    ['', '70', 'B'], // 키가 비었다
    ['175', '500', 'A'], // 몸무게가 범위 밖
  ],
}

describe('usableRows', () => {
  it('특성의 범위 밖 행을 뺀다', () => {
    expect(usableRows(table, ['키'], '반', 'mean', { 키: { max: 200 } })).toEqual([0, 1, 3, 4])
  })

  it('빈 칸은 범위 검사를 안 받는다 - 결측치 설정이 다룬다', () => {
    expect(usableRows(table, ['키'], '반', 'mean', { 키: { min: 160 } })).toContain(3)
  })

  it('타깃에도 걸린다', () => {
    expect(usableRows(table, ['키'], '몸무게', 'mean', { 몸무게: { max: 100 } })).toEqual([
      0, 1, 2, 3,
    ])
  })

  it('특성에서 뺀 열의 범위는 아무 행도 안 뺀다', () => {
    expect(usableRows(table, ['키'], '반', 'mean', { 몸무게: { max: 100 } })).toEqual([
      0, 1, 2, 3, 4,
    ])
  })

  it('범위가 없으면 지금까지와 같다', () => {
    expect(usableRows(table, ['키'], '반', 'mean', undefined)).toEqual([0, 1, 2, 3, 4])
  })

  it('없는 열 이름은 조용히 지나간다 - 데이터를 바꾼 뒤에 남은 숫자다', () => {
    expect(usableRows(table, ['키', '없는 열'], '반', 'mean', { '없는 열': { max: 0 } })).toEqual([
      0, 1, 2, 3, 4,
    ])
  })
})

describe('화면의 행 수가 같은 규칙을 본다', () => {
  it('trainableRowCount가 범위로 빠진 행을 센다', () => {
    expect(trainableRowCount(table, ['키'], '반', 'mean', { 키: { max: 200 } }, undefined)).toBe(4)
  })

  it('rowUsage가 빠진 행을 말한다', () => {
    expect(rowUsage(table, ['키'], '반', 'mean', { 키: { max: 200 } })).toEqual({
      total: 5,
      usable: 4,
      dropped: 1,
    })
  })
})

describe('학습 계획', () => {
  /** 키 150~199와 300. 반이 번갈아 온다 — 층화가 성립하게. */
  function heights(count: number, extra: string[][] = []): Dataset {
    return {
      columns: ['키', '반'],
      rows: [
        ...Array.from({ length: count }, (_, index) => [
          String(150 + (index % 50)),
          index % 2 === 0 ? 'A' : 'B',
        ]),
        ...extra,
      ],
    }
  }

  function settings(
    preprocessing: TabularSettings['preprocessing'],
    method: 'holdout' | 'provided',
  ) {
    return {
      split: { method, testSize: 0.3, stratify: false, randomState: 42 },
      runtime: 'mljs',
      selectedAlgorithms: [{ algorithm: 'decision_tree' }],
      hyperparameters: {},
      data: { features: ['키'], target: '반', preprocessing },
    } satisfies Settings
  }

  const RANGE: TabularSettings['preprocessing'] = {
    missing: 'mean',
    scaling: 'none',
    categoricalEncoding: 'onehot',
    outliers: 'range',
    ranges: { 키: { max: 200 } },
  }

  it('범위 밖 행이 훈련에도 테스트에도 안 들어간다', () => {
    const plan = planRun({
      dataset: heights(40, [
        ['300', 'A'],
        ['310', 'B'],
      ]),
      testDataset: null,
      settings: settings(RANGE, 'holdout'),
      taskType: 'classification',
    })
    expect(plan.ok).toBe(true)
    if (!plan.ok) return
    expect(plan.usable).toHaveLength(40)
    expect([...plan.split.trainIndices, ...plan.split.testIndices]).not.toContain(40)
    expect([...plan.split.trainIndices, ...plan.split.testIndices]).not.toContain(41)
  })

  it('따로 받은 테스트 데이터도 같은 범위로 뺀다', () => {
    const plan = planRun({
      dataset: heights(40),
      testDataset: heights(10, [['999', 'A']]),
      settings: settings(RANGE, 'provided'),
      taskType: 'classification',
    })
    expect(plan.ok).toBe(true)
    if (!plan.ok) return
    expect(plan.split.testIndices).toHaveLength(10)
    expect(plan.split.testIndices).not.toContain(10)
  })

  it('다른 방식을 고르면 적어 둔 범위가 행을 안 뺀다', () => {
    const plan = planRun({
      dataset: heights(40, [['300', 'A']]),
      testDataset: null,
      settings: settings({ ...RANGE, outliers: 'none' }, 'holdout'),
      taskType: 'classification',
    })
    expect(plan.ok).toBe(true)
    if (!plan.ok) return
    expect(plan.usable).toHaveLength(41)
  })
})

describe('입력칸 읽기', () => {
  it('비었으면 제한 없음이다', () => {
    expect(parseBound('  ')).toBeUndefined()
  })

  it('숫자를 읽는다 - 천 단위 쉼표와 소수와 음수까지', () => {
    expect(parseBound('1,000')).toBe(1000)
    expect(parseBound('-2.5')).toBe(-2.5)
  })

  it('숫자가 아니면 null이다 - 비운 것과 다르다', () => {
    expect(parseBound('abc')).toBeNull()
  })
})

describe('참고값', () => {
  it('전체 데이터의 최솟값·최댓값과 상자그림의 경계다', () => {
    const numbers: Dataset = {
      columns: ['x'],
      rows: [['1'], ['2'], ['3'], ['4'], ['5'], ['100']],
    }
    const guide = rangeGuide(numbers, 'x')
    expect(guide?.min).toBe(1)
    expect(guide?.max).toBe(100)
    expect(guide?.bounds).toEqual(boxPlot([1, 2, 3, 4, 5, 100])?.bounds)
  })

  it('숫자가 없는 열은 null이다', () => {
    expect(rangeGuide(table, '반')).toBeNull()
  })
})
