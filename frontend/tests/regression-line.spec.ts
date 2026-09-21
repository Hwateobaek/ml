/**
 * 회귀선 그림의 재료 (`ml/regression-line.ts`).
 *
 * **진짜 입구로 재현한다** — `parameters.spec.ts`와 같은 자리다. 전처리기를 손으로
 * 조립하지 않고 `fitPreprocessor` → `transform` → `fit` → 직렬화까지 실제로 지나간 뒤,
 * 파일 바이트에서 그림을 세운다. 이 파일이 지키려는 것이 **축과 값이 맞물리는가**라서
 * 중간을 건너뛰면 그 맞물림을 안 지나간다.
 *
 * **계수가 맞는지는 여기서 안 본다** (`sklearn-parity.spec.ts`의 몫이다). 여기 있는 것은
 * "그 계수로 그린 선이 점과 같은 좌표계에 서는가"다.
 */

import { describe, expect, it } from 'vitest'

import en from '../src/locales/en.json'
import ko from '../src/locales/ko.json'
import { fit } from '../src/ml/engines/mljs'
import { LINEAR_V2_FORMAT, TREE_FORMAT, parseLinearRegression } from '../src/ml/models'
import {
  fitPreprocessor,
  targetValues,
  transform,
  type Dataset,
  type Preprocessor,
} from '../src/ml/preprocess'
import {
  REGRESSION_TEXT,
  regressionChartFor,
  showsRegressionLine,
  type RegressionChartKind,
} from '../src/ml/regression-line'
import type { Experiment, Preprocessing } from '../src/project/schema'

/**
 * 몸무게가 키에 거의 붙어 있는 표. **마지막 줄의 타깃이 비어 있다** — 수로 안 읽히는
 * 값이 그림에서 어떻게 되는지를 볼 자리가 필요하다.
 */
const DATASET: Dataset = {
  columns: ['키', '몸무게', '지역'],
  rows: [
    ['150', '41', '서울'],
    ['155', '45', '부산'],
    ['160', '52', '서울'],
    ['165', '56', '부산'],
    ['170', '63', '서울'],
    ['175', '67', '부산'],
    ['180', '74', '서울'],
    ['185', '78', '부산'],
    ['190', '', '서울'],
  ],
}

/** 타깃이 비지 않은 줄. 학습은 이것으로만 한다. */
const ROWS = [0, 1, 2, 3, 4, 5, 6, 7]
const ALL_ROWS = [...ROWS, 8]

/** 상한을 안 건드리는 값. 표본 검사만 작은 값을 따로 넘긴다. */
const NO_LIMIT = 1_000

function options(scaling: Preprocessing['scaling']): Preprocessing {
  return { missing: 'drop', scaling, categoricalEncoding: 'onehot' }
}

function settingsOf(
  features: readonly string[],
  target: string,
  scaling: Preprocessing['scaling'],
  rows: readonly number[] = ROWS,
): Experiment['settings'] {
  return {
    // 스냅샷 스키마에 맞춘다 (`clusters.spec.ts`와 같은 사정이다).
    data: { features: [...features], target, preprocessing: options(scaling) },
    trainIndices: rows,
    split: { randomState: 42 },
  } as unknown as Experiment['settings']
}

interface Trained {
  readonly preprocessor: Preprocessor
  readonly bytes: Uint8Array
  readonly format: string
}

/** 학습부터 직렬화까지 실제로 지나간다. 돌려주는 것은 `.mlpx`에 담기는 바이트다. */
async function trained(
  features: readonly string[],
  target: string,
  scaling: Preprocessing['scaling'] = 'none',
): Promise<Trained> {
  const preprocessing = options(scaling)
  const preprocessor = fitPreprocessor(DATASET, ROWS, features, preprocessing)
  const matrix = transform(preprocessor, DATASET, ROWS, preprocessing.categoricalEncoding)
  const { model } = await fit('linear_regression', {
    features: matrix,
    rowIndices: ROWS,
    target: targetValues(DATASET, ROWS, target),
    taskType: 'regression',
    hyperparameters: {},
    randomState: 42,
  })
  return {
    preprocessor,
    bytes: new TextEncoder().encode(JSON.stringify(model)),
    format: (model as { format: string }).format,
  }
}

/** 원본 표의 값 하나. 그림의 좌표가 이것과 같아야 한다. */
function cell(row: number, column: string): number {
  return Number(DATASET.rows[row]?.[DATASET.columns.indexOf(column)] ?? '')
}

describe('회귀선 그림', () => {
  it('특성이 하나면 그 특성이 가로축이고 세로축이 타깃이다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(['키'], '몸무게', 'none'),
      NO_LIMIT,
    )

    expect(chart?.kind).toBe('feature')
    expect(chart?.featureName).toBe('키')
    expect(chart?.targetName).toBe('몸무게')
    // **점은 학생의 데이터 그대로다.** 전처리된 값이 아니라 원래 단위다.
    expect(chart?.points.map((point) => point.x)).toEqual(ROWS.map((row) => cell(row, '키')))
    expect(chart?.points.map((point) => point.y)).toEqual(ROWS.map((row) => cell(row, '몸무게')))
    expect(chart?.points.map((point) => point.row)).toEqual(ROWS)
  })

  it('선이 표의 계수·절편과 같은 선이다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(['키'], '몸무게', 'none'),
      NO_LIMIT,
    )

    // **모델이 배운 값 표가 보여주는 그 수로 다시 세워 본다** (`ml/parameters.ts`).
    // 여기가 갈리면 화면은 자기가 적어 놓은 식과 다른 선을 그린다.
    const model = parseLinearRegression(JSON.parse(new TextDecoder().decode(bytes)))
    const at = (x: number): number => (model.coefficients[0] ?? 0) * x + model.intercept

    const [from, to] = chart?.line ?? []
    expect(chart?.line).toHaveLength(2)
    // 선은 전체 행의 가로 범위를 지난다.
    expect(from?.x).toBe(150)
    expect(to?.x).toBe(185)
    expect(from?.y).toBeCloseTo(at(from?.x ?? 0), 9)
    expect(to?.y).toBeCloseTo(at(to?.x ?? 0), 9)
    // 선이 실제로 기울어져 있다 — 수평선이면 위 둘은 아무것도 안 말한다.
    expect(to?.y ?? 0).toBeGreaterThan(from?.y ?? 0)
  })

  it('스케일링을 켜도 눈금은 학생의 값이다', async () => {
    // 축에 붙는 것이 학생의 열 이름이므로 눈금도 학생의 값이어야 한다. 되돌리지 않으면
    // `키`라고 적힌 축에 `-1.4`가 선다.
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게', 'standard')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(['키'], '몸무게', 'standard'),
      NO_LIMIT,
    )

    expect(chart?.points.map((point) => point.x)).toEqual(ROWS.map((row) => cell(row, '키')))
    expect(chart?.line[0]?.x).toBeCloseTo(150, 9)
  })

  it('특성이 여럿이면 예측 대 실제이고 선은 대각선이다', async () => {
    const features = ['키', '지역']
    const { format, bytes, preprocessor } = await trained(features, '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(features, '몸무게', 'none'),
      NO_LIMIT,
    )

    expect(chart?.kind).toBe('fit')
    // **가로축이 우리 어휘라 열 이름이 없다.** 화면이 그 자리에 `실제 값`을 쓴다.
    expect(chart?.featureName).toBeNull()
    expect(chart?.points.map((point) => point.x)).toEqual(ROWS.map((row) => cell(row, '몸무게')))
    // 예측이 실제와 아주 다르지는 않다 — 그래야 이 그림이 뜻을 갖는다.
    for (const point of chart?.points ?? []) expect(Math.abs(point.y - point.x)).toBeLessThan(5)

    const [from, to] = chart?.line ?? []
    expect(from?.x).toBe(from?.y)
    expect(to?.x).toBe(to?.y)
    expect(to?.x ?? 0).toBeGreaterThan(from?.x ?? 0)
  })

  it('범주 열 하나짜리 모델도 예측 대 실제다', async () => {
    // 원핫이면 칸이 둘이라 특성이 하나가 아니고, 0과 1 두 값 위의 선은 읽을 것이 없다.
    const { format, bytes, preprocessor } = await trained(['지역'], '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(['지역'], '몸무게', 'none'),
      NO_LIMIT,
    )

    expect(chart?.kind).toBe('fit')
    expect(chart?.featureName).toBeNull()
  })

  it('수로 안 읽히는 타깃은 그림에서 빠진다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      // 빈 칸이 든 줄까지 그리라고 시킨다 — 데이터가 바뀐 파일이 그렇다.
      settingsOf(['키'], '몸무게', 'none', ALL_ROWS),
      NO_LIMIT,
    )

    expect(chart?.total).toBe(ROWS.length)
    expect(chart?.points.map((point) => point.row)).toEqual(ROWS)
  })

  it('그릴 줄이 한 줄도 없으면 안 그린다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    expect(
      regressionChartFor(
        format,
        bytes,
        DATASET,
        preprocessor,
        settingsOf(['키'], '몸무게', 'none', [8]),
        NO_LIMIT,
      ),
    ).toBeNull()
  })

  it('가로 좌표가 전부 같으면 선을 안 그린다', async () => {
    // 선이 아니라 점이다. 두 끝점이 같은 자리면 Chart.js는 아무것도 안 그리는데,
    // 그때 범례에는 `회귀선`이 서 있다 — 없는 것을 이름으로 말하는 자리다.
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const chart = regressionChartFor(
      format,
      bytes,
      DATASET,
      preprocessor,
      settingsOf(['키'], '몸무게', 'none', [3, 3]),
      NO_LIMIT,
    )

    expect(chart?.points).toHaveLength(2)
    expect(chart?.line).toEqual([])
  })

  it('점이 상한을 넘으면 표본이고 그 사실이 드러난다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const draw = () =>
      regressionChartFor(
        format,
        bytes,
        DATASET,
        preprocessor,
        settingsOf(['키'], '몸무게', 'none'),
        3,
      )

    const chart = draw()
    expect(chart?.drawn).toBe(3)
    expect(chart?.total).toBe(ROWS.length)
    // **같은 씨앗이면 같은 표본이다.** 매번 다른 점을 그리면 학생은 자기가 뭘 바꿔서
    // 그림이 바뀐 줄 안다.
    expect(draw()?.points.map((point) => point.row)).toEqual(
      chart?.points.map((point) => point.row),
    )
    // 뽑은 뒤 원래 순서로 되돌린다 — 그리는 차례가 표본 뽑기로 흔들릴 이유가 없다.
    const rows = chart?.points.map((point) => point.row) ?? []
    expect([...rows].sort((a, b) => a - b)).toEqual(rows)
  })

  it('선형 회귀가 아니면 선이 없다', () => {
    // 나무도 이웃도 회귀를 하지만 그 예측은 직선이 아니다. 형식으로 판정한다.
    expect(showsRegressionLine(TREE_FORMAT)).toBe(false)
    expect(showsRegressionLine(LINEAR_V2_FORMAT)).toBe(false)
    expect(showsRegressionLine(undefined)).toBe(false)
  })

  it('재료가 하나라도 없으면 안 그린다', async () => {
    const { format, bytes, preprocessor } = await trained(['키'], '몸무게')
    const settings = settingsOf(['키'], '몸무게', 'none')

    // 모델을 뺀 채로 받은 파일.
    expect(
      regressionChartFor(format, undefined, DATASET, preprocessor, settings, NO_LIMIT),
    ).toBeNull()
    // 데이터를 뺀 채로 받은 파일.
    expect(regressionChartFor(format, bytes, null, preprocessor, settings, NO_LIMIT)).toBeNull()
    // 전처리기가 안 담긴 파일.
    expect(regressionChartFor(format, bytes, DATASET, null, settings, NO_LIMIT)).toBeNull()
    // 다른 형식으로 담긴 모델.
    expect(
      regressionChartFor(TREE_FORMAT, bytes, DATASET, preprocessor, settings, NO_LIMIT),
    ).toBeNull()
    // 남이 편집해 깨진 파일.
    expect(
      regressionChartFor(
        format,
        new TextEncoder().encode('{'),
        DATASET,
        preprocessor,
        settings,
        NO_LIMIT,
      ),
    ).toBeNull()
    // **다른 전처리기로 배운 모델** — 계수 수와 칸 수가 어긋나면 해석기가 던진다.
    const narrow = fitPreprocessor(DATASET, ROWS, ['키', '지역'], options('none'))
    expect(regressionChartFor(format, bytes, DATASET, narrow, settings, NO_LIMIT)).toBeNull()
  })

  it('종류마다의 문구가 두 언어에 다 있다', () => {
    const flat = (tree: Record<string, unknown>, prefix = ''): Set<string> => {
      const out = new Set<string>()
      for (const [key, value] of Object.entries(tree)) {
        const path = prefix ? `${prefix}.${key}` : key
        if (typeof value === 'string') out.add(path)
        else if (value && typeof value === 'object') {
          for (const inner of flat(value as Record<string, unknown>, path)) out.add(inner)
        }
      }
      return out
    }
    const korean = flat(ko as unknown as Record<string, unknown>)
    const english = flat(en as unknown as Record<string, unknown>)
    const kinds: RegressionChartKind[] = ['feature', 'fit']

    for (const kind of kinds) {
      const text = REGRESSION_TEXT[kind]
      for (const key of [text.title, text.lead, text.line, text.axisX, text.axisY]) {
        if (key === null) continue
        expect(korean.has(key), `${key} (ko)`).toBe(true)
        expect(english.has(key), `${key} (en)`).toBe(true)
      }
    }
    // **두 종류가 서로 다른 문구를 든다.** 같은 키를 들면 갈라 놓은 뜻이 없다.
    expect(REGRESSION_TEXT.feature.title).not.toBe(REGRESSION_TEXT.fit.title)
    expect(REGRESSION_TEXT.feature.lead).not.toBe(REGRESSION_TEXT.fit.lead)
    expect(REGRESSION_TEXT.feature.line).not.toBe(REGRESSION_TEXT.fit.line)
  })
})
