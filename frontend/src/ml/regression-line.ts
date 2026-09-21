/**
 * 회귀선 그림의 재료 (`open-decisions.md` "선형 회귀는 회귀선을 그린다").
 *
 * **여기가 이 기능의 전부다. 화면은 받은 것을 그리기만 한다** (§8.3) —
 * `ml/clusters.ts`가 군집 산점도에 하는 일과 같은 자리이고, 점의 상한과 표본 뽑기도
 * 같은 것을 쓴다(`SCATTER_POINT_LIMIT`, `sampledIndices`).
 *
 * **그림이 둘이고 갈림은 특성 수가 정한다** (`RegressionChartKind`). 수치 특성 하나면
 * 교과서의 그 그림이고(가로축이 그 특성, 선이 모델), 여럿이면 **예측 대 실제**다 —
 * 가로축 하나로 여러 특성의 모델을 정직하게 못 그리기 때문이고, 그렇게 그린 선은
 * 점들과 나란히 서지 않아 **학생이 "모델이 틀렸다"로 읽는다.**
 *
 * **선을 따로 계산하지 않는다.** 두 끝점의 세로 좌표는 **점을 찍을 때 쓴 그 예측값**이다.
 * 식을 한 번 더 적으면 그 식이 `ml/parameters.ts`가 보여주는 계수와 갈리는 날이 오고,
 * 그때 화면은 자기가 적어 놓은 식과 다른 선을 그린다.
 *
 * **어느 형식이 선을 갖는지는 이 파일이 안다. 화면은 모른다** (§9.1) —
 * `ml/parameters.ts`·`ml/loss-curve.ts`와 같은 규칙이다.
 */

import { dataSnapshot, type Experiment } from '../project/schema'
import { LINEAR_REGRESSION_FORMAT, loadLinearRegressionModel } from './models'
import {
  targetValues,
  toNumber,
  transform,
  unscale,
  type Dataset,
  type Preprocessor,
} from './preprocess'
import { sampledIndices } from './shuffle'

/**
 * 어떤 그림인가.
 *
 * - `feature` — 가로축이 그 특성, 세로축이 타깃. **선이 모델 그 자체다.**
 * - `fit` — 가로축이 실제 값, 세로축이 예측값. **선은 `예측 = 실제`인 대각선**이고
 *   모델이 아니라 **기준**이다. 점이 그 선에 붙을수록 잘 맞힌 것이다.
 */
export type RegressionChartKind = 'feature' | 'fit'

/** 그림 한 종류의 문구 키 묶음. */
export interface RegressionChartText {
  readonly title: string
  readonly lead: string
  /** 선의 범례 이름. `feature`는 모델 자체이고 `fit`은 기준선이라 다른 말이다. */
  readonly line: string
  /**
   * 축 이름의 키. **`null`이면 그 축은 학생의 열 이름이다** — `feature` 그림의 두 축이
   * 그렇고(특성 이름 · 타깃 이름), 그 이름은 우리 어휘가 아니라 번역하지 않는다.
   */
  readonly axisX: string | null
  readonly axisY: string | null
}

/**
 * 종류마다의 문구 키. **화면에서 삼항으로 고르지 않는다** (`ml/clusters.ts`의
 * `clusterSummaryLeadKey`와 같은 자리, `CLAUDE.md` §4).
 *
 * 두 갈래를 서로 바꿔도 화면 검사는 전부 통과한다 — 그때 특성 하나짜리 그림에
 * *"특성이 여럿이라"*로 시작하는 문단이 붙는다. **키를 통째로 적어 두는 이유**는
 * 조립한 키를 정적 `t()` 검사가 못 보기 때문이다(`ml/parameters.ts`의
 * `PARAMETER_TITLE_KEYS`와 같은 규칙).
 */
export const REGRESSION_TEXT: Readonly<Record<RegressionChartKind, RegressionChartText>> = {
  feature: {
    title: 'results.regressionLineTitle',
    lead: 'results.regressionLineLead',
    line: 'results.regressionFitLine',
    axisX: null,
    axisY: null,
  },
  fit: {
    title: 'results.regressionFitTitle',
    lead: 'results.regressionFitLead',
    line: 'results.regressionGuideLine',
    axisX: 'results.actual',
    axisY: 'results.regressionPredicted',
  },
}

/** 그림에 찍는 점 하나. `row`는 원본 표의 행 번호다. */
export interface RegressionPoint {
  readonly row: number
  readonly x: number
  readonly y: number
}

/** 선의 끝점. */
export interface LinePoint {
  readonly x: number
  readonly y: number
}

export interface RegressionChart {
  readonly kind: RegressionChartKind
  /**
   * 가로축이 되는 특성의 열 이름. **`fit` 그림에는 없다** — 거기서는 축 이름이
   * 학생의 열 이름이 아니라 우리 어휘(`실제 값`)라 화면이 번역해 쓴다.
   */
  readonly featureName: string | null
  /** 타깃 열 이름. 두 그림 다 이 이름을 축에 쓴다. */
  readonly targetName: string
  readonly points: readonly RegressionPoint[]
  /**
   * 선의 두 끝점. **비어 있으면 선을 안 그린다** — 점이 없거나 가로 좌표가 전부 같아
   * 선이 될 수 없을 때다.
   */
  readonly line: readonly LinePoint[]
  /** 실제로 찍는 점 수. */
  readonly drawn: number
  /** 그림이 본 전체 행 수. **`drawn`과 다르면 화면이 그 사실을 말한다.** */
  readonly total: number
}

/**
 * 이 형식이 회귀선을 갖는가. **형식으로 판정한다. 알고리즘 이름이 아니다** —
 * 같은 알고리즘이 다른 형식으로 담길 수 있고(옛 파일), 우리가 읽을 수 있는지는 형식이
 * 답한다.
 *
 * **선형 회귀뿐이다.** 나무도 이웃도 회귀를 하지만 그 예측은 계단이나 조각이라
 * 직선 하나로 못 적는다 — 그 그림들은 각자의 결정이 필요하고, 여기서 함께 열면
 * **곡선을 직선으로 그리는 거짓말**이 된다.
 */
export function showsRegressionLine(format: string | undefined): boolean {
  return format === LINEAR_REGRESSION_FORMAT
}

/** 두 끝점. 같은 자리면 선이 아니므로 빈 배열이다. */
function endpoints(from: LinePoint, to: LinePoint): readonly LinePoint[] {
  return from.x === to.x ? [] : [from, to]
}

/**
 * 재료를 만든다. **못 만들면 `null`이고, 그때 화면은 아무것도 안 그린다**
 * (§9.2 "없는 것을 이름으로 말하지 않는다").
 *
 * `null`이 되는 경우는 넷이다 — 선을 안 갖는 형식 · 재료가 파일에 없거나(데이터를 뺀
 * 파일) · 읽다가 실패했거나(남이 편집한 파일) · 수로 읽히는 타깃 값이 한 줄도 없을 때.
 * **넷을 가르지 않는 이유는 화면이 할 일이 넷 다 같기 때문이다.**
 *
 * **훈련 데이터를 그린다.** 선이 그 행들에서 나왔고, 이 그림이 답하는 질문이 *"모델이
 * 무엇을 보고 이 선을 그었나"*다. 테스트 데이터에서의 성적은 위의 점수표가 말한다.
 *
 * @param limit 찍을 점의 상한. 넘으면 `randomState`로 표본을 뽑는다.
 */
export function regressionChartFor(
  format: string | undefined,
  bytes: Uint8Array | undefined,
  dataset: Dataset | null,
  preprocessor: Preprocessor | null | undefined,
  settings: Experiment['settings'],
  limit: number,
): RegressionChart | null {
  if (!showsRegressionLine(format) || !bytes || !dataset || !preprocessor) return null

  try {
    // 이 그림은 표 전용이다 — 등록부가 그렇게 세워 두었다 (`ml/metric-panels.ts`).
    const snapshot = dataSnapshot('tabular', settings)
    // **타깃을 못 읽으면 안 그린다.** 세로축에 설 값이 없다 — 회귀가 타깃 없이 학습될
    // 길은 없으므로 여기 오는 것은 남이 편집한 파일이다.
    const target = snapshot.target
    if (target === undefined) return null

    const encoding = snapshot.preprocessing.categoricalEncoding
    const rows = settings.trainIndices
    const matrix = transform(preprocessor, dataset, rows, encoding)
    const predict = loadLinearRegressionModel(JSON.parse(new TextDecoder().decode(bytes)))
    // **예측은 한 번만 한다.** 점의 세로 좌표도 선의 끝점도 이 배열에서 나온다.
    const predicted = predict(matrix).map((value) => Number(value))
    const actual = targetValues(dataset, rows, target).map(toNumber)

    /**
     * **수로 안 읽히는 타깃은 그림에서 뺀다.** 회귀는 수치 타깃에서만 학습되므로
     * 정상 경로에는 없고, 여기 오는 것은 데이터가 바뀐 파일이다 — 그때 `NaN` 점을
     * 찍으면 축이 통째로 망가진다.
     */
    const usable: number[] = []
    for (let i = 0; i < rows.length; i += 1) {
      if (actual[i] !== null && Number.isFinite(predicted[i] ?? Number.NaN)) usable.push(i)
    }
    if (usable.length === 0) return null

    /**
     * **수치 특성 하나짜리 모델인가.** 원핫으로 늘어난 칸이 있으면 `featureNames`가
     * 여럿이라 여기서 갈린다 — 범주 열 하나뿐인 모델도 `fit` 쪽이다(0/1 두 값 위의
     * 선은 읽을 것이 없다).
     */
    const only = preprocessor.columns.length === 1 ? preprocessor.columns[0] : undefined
    const single =
      only && only.kind === 'numeric' && preprocessor.featureNames.length === 1 ? only : undefined

    const coordinates = usable.map((i) => ({
      row: rows[i] as number,
      // **원래 단위로 되돌린다** — 축에 붙는 것이 학생의 열 이름이므로 눈금도 학생의
      // 값이어야 한다 (군집 산점도와 같은 자리).
      x: single ? unscale(single, matrix[i]?.[0] ?? 0) : (actual[i] as number),
      y: single ? (actual[i] as number) : (predicted[i] as number),
      /** 선의 끝점을 고르는 데만 쓴다. `feature` 그림에서 그 x의 모델 값이다. */
      fitted: predicted[i] as number,
    }))

    /**
     * **선은 전체 행의 범위를 지난다.** 표본을 뽑았어도 모델이 배운 범위는 그대로이고,
     * 그 범위를 표본이 좁히면 학생이 보는 선이 데이터마다 다른 길이가 된다.
     */
    let low = coordinates[0] as (typeof coordinates)[number]
    let high = low
    for (const point of coordinates) {
      if (point.x < low.x) low = point
      if (point.x > high.x) high = point
    }

    const line = single
      ? endpoints({ x: low.x, y: low.fitted }, { x: high.x, y: high.fitted })
      : diagonal(coordinates)

    const total = coordinates.length
    const picked =
      total <= limit
        ? coordinates.map((_point, index) => index)
        : sampledIndices(total, limit, settings.split.randomState)
    const points = picked.map((index) => {
      const point = coordinates[index] as (typeof coordinates)[number]
      return { row: point.row, x: point.x, y: point.y }
    })

    return {
      kind: single ? 'feature' : 'fit',
      featureName: single ? single.name : null,
      targetName: target,
      points,
      line,
      drawn: points.length,
      total,
    }
  } catch {
    // 못 읽는 파일이다. 이유를 말하는 자리는 다른 곳이고, 여기서는 안 그린다.
    return null
  }
}

/**
 * `예측 = 실제` 대각선. **점 구름 전체를 덮는 한 선이라** 가로와 세로의 범위를 함께
 * 본다 — 한쪽 범위만 보면 선이 구름의 절반에서 끊긴다.
 */
function diagonal(
  coordinates: readonly { readonly x: number; readonly y: number }[],
): readonly LinePoint[] {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const point of coordinates) {
    min = Math.min(min, point.x, point.y)
    max = Math.max(max, point.x, point.y)
  }
  return endpoints({ x: min, y: min }, { x: max, y: max })
}
