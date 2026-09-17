/**
 * `data/visualize.ts`의 계산.
 *
 * **눈으로는 안 보이는 것만 여기서 지킨다** — 구간 경계에 걸친 값이 어느 칸에 드는지,
 * 결측이 있는 열에서 행이 어긋나지 않는지, 상수 열에서 NaN이 새지 않는지. 셋 다 화면에
 * 그려 놓으면 그럴듯해 보이고, 틀린 줄은 한참 뒤에 안다.
 */

import { describe, expect, it } from 'vitest'

import {
  boxPlot,
  categoryCounts,
  crossTab,
  defaultLineAxis,
  groupedBoxPlots,
  correlation,
  correlationMatrix,
  descriptiveStats,
  fiveNumbers,
  histogram,
  lineChart,
  numericPairs,
  numericValues,
  outlierCounts,
} from '@/data/visualize'
import { OUTLIER_MARK_COUNT } from '@/limits'
import { outlierBounds } from '@/ml/outliers'
import type { Dataset } from '@/ml/preprocess'

const dataset: Dataset = {
  columns: ['a', 'b', 'label'],
  rows: [
    ['1', '2', 'x'],
    ['2', '4', 'y'],
    ['3', '6', 'x'],
    ['', '8', 'y'],
    ['5', '', 'x'],
  ],
}

describe('numericValues', () => {
  it('빈 칸을 0으로 채우지 않고 뺀다', () => {
    expect(numericValues(dataset, 'a')).toEqual([1, 2, 3, 5])
  })

  it('없는 열은 빈 배열이다', () => {
    expect(numericValues(dataset, 'nope')).toEqual([])
  })

  it('숫자가 아닌 값은 뺀다', () => {
    expect(numericValues(dataset, 'label')).toEqual([])
  })
})

describe('numericPairs', () => {
  it('한쪽이라도 비면 그 행을 버려 행이 어긋나지 않는다', () => {
    expect(numericPairs(dataset, 'a', 'b')).toEqual([
      { x: 1, y: 2, row: 0 },
      { x: 2, y: 4, row: 1 },
      { x: 3, y: 6, row: 2 },
    ])
  })
})

describe('histogram', () => {
  it('값이 없으면 칸도 없다', () => {
    expect(histogram([])).toEqual([])
  })

  it('값이 전부 같으면 칸 하나에 전부 담는다', () => {
    expect(histogram([7, 7, 7])).toEqual([{ start: 7, end: 7, count: 3 }])
  })

  it('모든 값이 어느 칸엔가 정확히 한 번 들어간다', () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const bins = histogram(values, 5)
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(values.length)
  })

  it('최댓값은 마지막 칸에 든다 - 경계 밖으로 새지 않는다', () => {
    const bins = histogram([0, 10], 5)
    expect(bins[bins.length - 1]?.count).toBe(1)
    expect(bins[0]?.count).toBe(1)
  })
})

describe('fiveNumbers', () => {
  it('값이 없으면 null이다', () => {
    expect(fiveNumbers([])).toBeNull()
  })

  it('중앙값과 사분위수를 낸다', () => {
    const summary = fiveNumbers([1, 2, 3, 4, 5])
    expect(summary).toEqual({ min: 1, q1: 2, median: 3, q3: 4, max: 5 })
  })

  it('사이에 있는 분위수는 선형으로 잇는다', () => {
    expect(fiveNumbers([1, 2, 3, 4])?.median).toBe(2.5)
  })
})

describe('descriptiveStats', () => {
  it('값이 없으면 null이다', () => {
    expect(descriptiveStats([])).toBeNull()
  })

  it('pandas describe()와 같은 여덟 칸을 낸다 — 표준편차는 n−1로 나눈다', () => {
    const stats = descriptiveStats([2, 4, 4, 4, 5, 5, 7, 9])
    expect(stats?.count).toBe(8)
    expect(stats?.mean).toBe(5)
    // 편차 제곱합 32를 7로 나눈다. n으로 나누면 정확히 2가 된다.
    expect(stats?.std).toBeCloseTo(Math.sqrt(32 / 7), 12)
    expect(stats).toMatchObject({ min: 2, q1: 4, median: 4.5, q3: 5.5, max: 9 })
  })

  it('값이 하나면 표준편차가 없다', () => {
    expect(descriptiveStats([3])).toMatchObject({ count: 1, mean: 3, std: null })
  })
})

describe('correlation', () => {
  it('완전히 같이 커지면 1이다', () => {
    expect(correlation([1, 2, 3], [2, 4, 6])).toBeCloseTo(1)
  })

  it('반대로 움직이면 -1이다', () => {
    expect(correlation([1, 2, 3], [6, 4, 2])).toBeCloseTo(-1)
  })

  it('상수 열에서 NaN을 흘리지 않는다', () => {
    expect(correlation([1, 1, 1], [1, 2, 3])).toBe(0)
  })

  it('값이 둘 미만이면 0이다', () => {
    expect(correlation([1], [2])).toBe(0)
  })
})

describe('correlationMatrix', () => {
  it('대각선은 1이다', () => {
    const matrix = correlationMatrix(dataset, ['a', 'b'])
    expect(matrix[0]?.[0]).toBe(1)
    expect(matrix[1]?.[1]).toBe(1)
  })

  it('대칭이다', () => {
    const matrix = correlationMatrix(dataset, ['a', 'b'])
    expect(matrix[0]?.[1]).toBeCloseTo(matrix[1]?.[0] ?? 0)
  })

  it('결측이 있어도 짝지은 행끼리만 재므로 유한하다', () => {
    const matrix = correlationMatrix(dataset, ['a', 'b'])
    for (const row of matrix) {
      for (const cell of row) expect(Number.isFinite(cell)).toBe(true)
    }
  })
})

describe('boxPlot', () => {
  it('값이 없으면 null이다', () => {
    expect(boxPlot([])).toBeNull()
  })

  it('경계가 전처리기와 같은 함수에서 나온다', () => {
    const values = [1, 2, 3, 4, 5, 100]
    expect(boxPlot(values)?.bounds).toEqual(outlierBounds(values))
  })

  it('수염은 경계가 아니라 경계 안에서 가장 먼 실제 값이다', () => {
    const plot = boxPlot([1, 2, 3, 4, 5, 100])
    expect(plot?.whiskerLow).toBe(1)
    expect(plot?.whiskerHigh).toBe(5)
    expect(plot?.outlierCount).toBe(1)
    expect(plot?.marks).toEqual([100])
  })

  it('같은 이상치는 점 하나로 찍고 개수는 전부 센다', () => {
    // 안쪽 값이 넉넉해야 한다 — 100이 여덟 개 중 셋이면 Q3가 100이 되어 이상치가 아니다.
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 100, 100]
    const plot = boxPlot(values)
    expect(plot?.marks).toEqual([100])
    expect(plot?.outlierCount).toBe(3)
  })

  it('점의 수에 상한이 있다', () => {
    const inside = Array.from({ length: 2000 }, () => 0).concat(
      Array.from({ length: 2000 }, () => 1),
    )
    const far = Array.from({ length: OUTLIER_MARK_COUNT + 50 }, (_, index) => 1000 + index)
    const plot = boxPlot([...inside, ...far])
    expect(plot?.marks.length).toBe(OUTLIER_MARK_COUNT)
    expect(plot?.outlierCount).toBe(OUTLIER_MARK_COUNT + 50)
  })

  it('IQR이 0이면 경계도 이상치도 없다', () => {
    const plot = boxPlot([0, 0, 0, 0, 1])
    expect(plot?.bounds).toBeNull()
    expect(plot?.outlierCount).toBe(0)
  })
})

describe('outlierCounts', () => {
  const table: Dataset = {
    columns: ['x', 'label'],
    rows: [
      ['1', 'a'],
      ['2', 'b'],
      ['3', 'a'],
      ['4', 'b'],
      ['5', 'a'],
      ['100', 'b'],
    ],
  }

  it('수치 열의 이상치를 전체 행으로 센다', () => {
    expect(outlierCounts(table, ['x']).get('x')).toBe(1)
  })

  it('숫자가 없는 열은 0이 아니라 목록에서 빠진다', () => {
    expect(outlierCounts(table, ['label']).has('label')).toBe(false)
  })
})

describe('범주 그림', () => {
  const table: Dataset = {
    columns: ['species', 'island', 'mass'],
    rows: [
      ['Adelie', 'Torgersen', '3700'],
      ['Gentoo', 'Biscoe', '5000'],
      ['Adelie', 'Biscoe', '3500'],
      ['Adelie', '', '3900'],
      ['', 'Dream', '3600'],
      ['Gentoo', 'Biscoe', ''],
    ],
  }

  it('categoryCounts는 빈 칸을 빼고 이름 차례로 센다', () => {
    expect(categoryCounts(table, 'species')).toEqual([
      { value: 'Adelie', count: 3 },
      { value: 'Gentoo', count: 2 },
    ])
  })

  it('categoryCounts는 숫자가 든 이름을 수의 크기로 놓는다', () => {
    const grades: Dataset = { columns: ['g'], rows: [['10'], ['2'], ['1'], ['2']] }
    expect(categoryCounts(grades, 'g').map((item) => item.value)).toEqual(['1', '2', '10'])
  })

  it('groupedBoxPlots는 범주가 비거나 수가 아닌 행을 뺀다', () => {
    const plots = groupedBoxPlots(table, 'mass', 'species')
    expect(plots.map((item) => item.group)).toEqual(['Adelie', 'Gentoo'])
    expect(plots[0]?.plot).toMatchObject({ min: 3500, median: 3700, max: 3900 })
    expect(plots[1]?.plot).toMatchObject({ min: 5000, max: 5000 })
  })

  it('crossTab은 안 나온 조합도 0으로 칸을 둔다', () => {
    expect(crossTab(table, 'island', 'species')).toEqual({
      rows: ['Biscoe', 'Torgersen'],
      columns: ['Adelie', 'Gentoo'],
      counts: [
        [1, 2],
        [1, 0],
      ],
    })
  })

  it('없는 열이면 비어 있다', () => {
    expect(categoryCounts(table, 'nope')).toEqual([])
    expect(groupedBoxPlots(table, 'mass', 'nope')).toEqual([])
    expect(crossTab(table, 'nope', 'species').rows).toEqual([])
  })
})

describe('lineChart', () => {
  it('가로축 값의 크기 차례로 놓는다 — 파일의 차례가 아니다', () => {
    const table: Dataset = {
      columns: ['year', 'price'],
      rows: [
        ['2010', '3'],
        ['2002', '1'],
        ['2006', '2'],
      ],
    }
    expect(lineChart(table, 'year', ['price'])).toEqual({
      labels: ['2002', '2006', '2010'],
      series: [{ column: 'price', values: [1, 2, 3] }],
      averaged: false,
    })
  })

  it('같은 가로축 값은 평균을 내고 그 사실을 알린다', () => {
    const table: Dataset = {
      columns: ['year', 'a', 'b'],
      rows: [
        ['2020', '1', '10'],
        ['2020', '3', ''],
        ['2021', '5', 'x'],
        ['', '100', '100'],
      ],
    }
    const chart = lineChart(table, 'year', ['a', 'b'])
    expect(chart.labels).toEqual(['2020', '2021'])
    expect(chart.series).toEqual([
      { column: 'a', values: [2, 5] },
      { column: 'b', values: [10, null] },
    ])
    expect(chart.averaged).toBe(true)
  })

  it('숫자가 아닌 가로축은 숫자가 섞인 이름 차례다', () => {
    const table: Dataset = {
      columns: ['month', 'v'],
      rows: [
        ['10월', '3'],
        ['2월', '2'],
        ['1월', '1'],
      ],
    }
    expect(lineChart(table, 'month', ['v']).labels).toEqual(['1월', '2월', '10월'])
  })
})

describe('defaultLineAxis', () => {
  it('시간을 가리키는 이름의 열을 먼저 잡는다', () => {
    expect(defaultLineAxis(['품목', '가격', '연도'])).toBe('연도')
    expect(defaultLineAxis(['id', 'Year', 'price'])).toBe('Year')
  })

  it('그런 열이 없으면 첫 열이고, 열이 없으면 빈 문자열이다', () => {
    expect(defaultLineAxis(['a', 'b'])).toBe('a')
    expect(defaultLineAxis([])).toBe('')
  })
})
