/**
 * 이상치의 경계와 클리핑 (open-decisions.md "이상치는 훈련 데이터의 IQR로 클리핑한다").
 *
 * **여기서 지키는 것은 넷이다** — 경계를 훈련 데이터로만 구하는가, 스케일을 자른 뒤에
 * 구하는가, 채운 값을 안 자르는가, 옛 전처리기 파일을 계속 읽는가. 넷 다 화면에서는
 * 그럴듯한 숫자로 보이고, 틀리면 점수가 조용히 달라진다.
 */

import { describe, expect, it } from 'vitest'

import { OUTLIER_IQR_MULTIPLIER } from '@/limits'
import {
  clipValue,
  countOutliers,
  isOutlier,
  outlierBounds,
  quartilesOfSorted,
} from '@/ml/outliers'
import {
  fitPreprocessor,
  parsePreprocessor,
  PREPROCESSOR_FORMAT,
  transform,
  type Dataset,
} from '@/ml/preprocess'
import type { Preprocessing } from '@/project/schema'

describe('경계', () => {
  it('배수는 교과서의 1.5다', () => {
    expect(OUTLIER_IQR_MULTIPLIER).toBe(1.5)
  })

  it('사분위수를 numpy 기본과 같은 선형 보간으로 구한다', () => {
    expect(quartilesOfSorted([1, 2, 3, 4, 5])).toEqual({ q1: 2, median: 3, q3: 4 })
  })

  it('Q1 − 1.5×IQR과 Q3 + 1.5×IQR이다', () => {
    // Q1 2, Q3 4, IQR 2 → [-1, 7]
    expect(outlierBounds([1, 2, 3, 4, 5])).toEqual({ low: -1, high: 7 })
  })

  it('정렬되지 않은 값도 같은 경계를 낸다', () => {
    expect(outlierBounds([5, 1, 4, 2, 3])).toEqual(outlierBounds([1, 2, 3, 4, 5]))
  })

  it('값이 없으면 경계가 없다', () => {
    expect(outlierBounds([])).toBeNull()
  })

  it('IQR이 0이면 경계가 없다 - 자르면 열이 한 값이 된다', () => {
    expect(outlierBounds([0, 0, 0, 0, 0, 0, 1])).toBeNull()
  })

  it('경계값 자체는 이상치가 아니다', () => {
    const bounds = { low: -1, high: 7 }
    expect(isOutlier(7, bounds)).toBe(false)
    expect(isOutlier(7.01, bounds)).toBe(true)
  })

  it('경계 밖을 가장 가까운 경계로 민다', () => {
    const bounds = { low: -1, high: 7 }
    expect(clipValue(100, bounds)).toBe(7)
    expect(clipValue(-50, bounds)).toBe(-1)
    expect(clipValue(3, bounds)).toBe(3)
  })

  it('경계가 없으면 그대로다', () => {
    expect(clipValue(100, null)).toBe(100)
  })

  it('이상치를 센다', () => {
    expect(countOutliers([1, 2, 3, 4, 5, 100])).toBe(1)
  })

  it('큰 배열을 인자로 안 펼친다 - 행 수만큼 와도 죽지 않는다', () => {
    const values = Array.from({ length: 200_000 }, (_, index) => index % 97)
    expect(() => countOutliers(values)).not.toThrow()
  })
})

const CLIP: Preprocessing = {
  missing: 'mean',
  scaling: 'standard',
  categoricalEncoding: 'onehot',
  outliers: 'clip',
}

/** 훈련 행 0~4는 1~5, 테스트 행 5는 아주 큰 값, 행 6은 빈 칸이다. */
const dataset: Dataset = {
  columns: ['x', 'label'],
  rows: [
    ['1', 'a'],
    ['2', 'a'],
    ['3', 'b'],
    ['4', 'b'],
    ['5', 'a'],
    ['1000', 'b'],
    ['', 'a'],
  ],
}
const TRAIN = [0, 1, 2, 3, 4]

describe('전처리기의 클리핑', () => {
  it('경계를 훈련 데이터로만 구한다 - 테스트의 큰 값이 경계를 안 넓힌다', () => {
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['x'], CLIP)
    expect(preprocessor.columns[0]?.clip).toEqual({ low: -1, high: 7 })
  })

  it('테스트 행도 훈련 경계로 자른다', () => {
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['x'], { ...CLIP, scaling: 'none' })
    expect(transform(preprocessor, dataset, [5], 'onehot')).toEqual([[7]])
  })

  it('스케일 기준을 자른 뒤의 값으로 구한다', () => {
    const withOutlier: Dataset = {
      columns: ['x'],
      rows: [['1'], ['2'], ['3'], ['4'], ['100']],
    }
    const clipped = fitPreprocessor(withOutlier, [0, 1, 2, 3, 4], ['x'], CLIP)
    const untouched = fitPreprocessor(withOutlier, [0, 1, 2, 3, 4], ['x'], {
      ...CLIP,
      outliers: 'none',
    })
    // 자르면 100이 7로 들어가 평균이 끌려가지 않는다.
    expect(clipped.columns[0]?.scale?.center).toBeLessThan(untouched.columns[0]?.scale?.center ?? 0)
  })

  it('대체값도 자른 뒤의 값으로 구한다', () => {
    const withOutlier: Dataset = {
      columns: ['x'],
      rows: [['1'], ['2'], ['3'], ['4'], ['100'], ['']],
    }
    const preprocessor = fitPreprocessor(withOutlier, [0, 1, 2, 3, 4], ['x'], {
      ...CLIP,
      scaling: 'none',
    })
    const bounds = preprocessor.columns[0]?.clip
    expect(bounds).toBeDefined()
    expect(preprocessor.columns[0]?.fill).toBeLessThanOrEqual(bounds?.high ?? 0)
  })

  it('채운 값은 안 자른다 - 0으로 채움을 고르면 0이 들어간다', () => {
    const shifted: Dataset = {
      columns: ['x'],
      rows: [['10'], ['11'], ['12'], ['13'], ['14'], ['']],
    }
    const preprocessor = fitPreprocessor(shifted, [0, 1, 2, 3, 4], ['x'], {
      ...CLIP,
      missing: 'zero',
      scaling: 'none',
    })
    // 경계는 [8, 16]이라 0은 밖이지만, 학생이 고른 대체값이므로 그대로다.
    expect(preprocessor.columns[0]?.clip).toEqual({ low: 8, high: 16 })
    expect(transform(preprocessor, shifted, [5], 'onehot')).toEqual([[0]])
  })

  it('안 고르면 경계를 안 적는다', () => {
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['x'], { ...CLIP, outliers: 'none' })
    expect(preprocessor.columns[0]?.clip).toBeUndefined()
  })

  it('필드가 없는 옛 설정은 안 자른다', () => {
    const legacy: Preprocessing = {
      missing: 'mean',
      scaling: 'standard',
      categoricalEncoding: 'onehot',
    }
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['x'], legacy)
    expect(preprocessor.columns[0]?.clip).toBeUndefined()
  })

  it('범주 열은 안 자른다', () => {
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['label'], CLIP)
    expect(preprocessor.columns[0]?.clip).toBeUndefined()
  })

  it('IQR이 0인 열은 안 자른다', () => {
    const flat: Dataset = { columns: ['x'], rows: [['0'], ['0'], ['0'], ['0'], ['1']] }
    const preprocessor = fitPreprocessor(flat, [0, 1, 2, 3, 4], ['x'], CLIP)
    expect(preprocessor.columns[0]?.clip).toBeUndefined()
  })
})

describe('전처리기 파일', () => {
  it('새로 쓰는 형식은 v2다', () => {
    expect(PREPROCESSOR_FORMAT).toBe('mlpx-preprocess-v2')
    expect(fitPreprocessor(dataset, TRAIN, ['x'], CLIP).format).toBe('mlpx-preprocess-v2')
  })

  it('경계가 저장 왕복을 지난다', () => {
    const preprocessor = fitPreprocessor(dataset, TRAIN, ['x'], CLIP)
    const revived = parsePreprocessor(JSON.parse(JSON.stringify(preprocessor)))
    expect(revived.columns[0]?.clip).toEqual({ low: -1, high: 7 })
  })

  it('v1 파일을 계속 읽는다 - 이미 나간 파일이 있다', () => {
    const revived = parsePreprocessor({
      format: 'mlpx-preprocess-v1',
      columns: [{ name: 'x', kind: 'numeric' }],
      featureNames: ['x'],
    })
    expect(revived.columns[0]?.clip).toBeUndefined()
  })

  it('경계가 뒤집힌 파일은 거부한다', () => {
    expect(() =>
      parsePreprocessor({
        format: PREPROCESSOR_FORMAT,
        columns: [{ name: 'x', kind: 'numeric', clip: { low: 7, high: -1 } }],
        featureNames: ['x'],
      }),
    ).toThrow()
  })

  it('모르는 형식 이름은 거부한다', () => {
    expect(() =>
      parsePreprocessor({
        format: 'mlpx-preprocess-v9',
        columns: [{ name: 'x', kind: 'numeric' }],
        featureNames: ['x'],
      }),
    ).toThrow()
  })
})
