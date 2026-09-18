/**
 * `mlpx-dbscan-v1` — DBSCAN 군집 (mlpx-spec.md §5.14).
 *
 * **sklearn의 `DBSCAN`에는 `predict`가 없다.** 학습한 데이터에 번호를 붙이는 것이 전부이고,
 * 새 점이 어느 무리인지는 답하지 않는다. 그런데 이 도구에는 예측 화면이 있다 — 그래서
 * **DBSCAN의 정의를 그대로 새 점에 적용한다**: 반경(`eps`) 안에 핵심 점(core point)이
 * 있으면 **가장 가까운 핵심 점**의 무리이고, 없으면 잡음(`-1`)이다.
 *
 * **학습의 경계 점 규칙과 한 곳이 다르다.** 학습에서 경계 점은 **먼저 닿은 무리**에 붙고
 * (`engines/dbscan.ts`), 여기서는 **가장 가까운** 핵심 점의 무리다. 두 무리의 반경 안에
 * 함께 드는 경계 점 하나를 다시 넣으면 다른 번호가 나올 수 있다 — 핵심 점과 잡음은 언제나
 * 같다. 새 점에는 "먼저"가 없어서 거리로 가른다.
 *
 * **그래서 모델은 핵심 점과 그 번호다.** 훈련 행 번호가 아니라 좌표를 담는다 — 핵심 점만
 * 필요하고, 좌표를 담으면 `dataset/`이 없는 파일에서도 예측이 된다 (참조형과 갈리는 자리다).
 *
 * **동점이면 앞의 핵심 점이 이긴다.** 핵심 점은 학습이 훑은 행 차례로 담긴다.
 *
 * **답은 문자열 번호다** — K-평균과 같다. 잡음은 `'-1'`이고 화면이 그것을 "잡음"으로 읽는다.
 */

import { z } from 'zod'

import type { ModelFile, Predict } from './types'
import { invalidModel } from './regression-tree'

export const DBSCAN_FORMAT = 'mlpx-dbscan-v1'

/** 어느 무리에도 속하지 않은 점의 번호. sklearn `labels_`의 `-1`과 같다. */
export const NOISE = -1

export interface DbscanModel extends ModelFile {
  readonly format: typeof DBSCAN_FORMAT
  readonly featureCount: number
  /** 반경. 이 거리 **이하**가 이웃이다 (sklearn과 같다). */
  readonly eps: number
  /** 무리 수. 잡음은 세지 않는다. */
  readonly clusterCount: number
  /** 핵심 점의 좌표. 학습이 훑은 행 차례다. */
  readonly cores: readonly (readonly number[])[]
  /** `cores[i]`의 무리 번호. 0부터 `clusterCount - 1`까지다. */
  readonly coreLabels: readonly number[]
}

const dbscanSchema = z.looseObject({
  format: z.literal(DBSCAN_FORMAT),
  featureCount: z.number(),
  eps: z.number(),
  clusterCount: z.number(),
  cores: z.array(z.array(z.number())),
  coreLabels: z.array(z.number()),
})

/** 예측 함수. **학습 쪽도 이 함수를 그대로 쓴다.** */
export function dbscanPredict(model: DbscanModel): Predict {
  const { featureCount, eps, clusterCount, cores, coreLabels } = model
  if (!Number.isInteger(featureCount) || featureCount <= 0) invalidModel('featureCount')
  if (!Number.isFinite(eps) || eps <= 0) invalidModel('eps')
  if (!Number.isInteger(clusterCount) || clusterCount < 0) invalidModel('clusterCount')
  if (cores.length !== coreLabels.length) invalidModel('coreLabels')
  const matrix = cores.map((core) => {
    if (core.length !== featureCount) invalidModel('cores')
    return Float64Array.from(core)
  })
  coreLabels.forEach((label) => {
    if (!Number.isInteger(label) || label < 0 || label >= clusterCount) invalidModel('coreLabels')
  })
  // 거리는 제곱으로 견준다 — 제곱근을 안 씌우면 반올림 차이가 없어 경계 판정이 정확하다.
  const reach = eps * eps

  return (features) =>
    features.map((row) => {
      if (row.length !== featureCount) invalidModel('featureCount')
      let best = NOISE
      let bestDistance = Number.POSITIVE_INFINITY
      matrix.forEach((core, index) => {
        let distance = 0
        for (let column = 0; column < featureCount; column += 1) {
          const gap = (row[column] ?? 0) - (core[column] ?? 0)
          distance += gap * gap
        }
        if (distance <= reach && distance < bestDistance) {
          bestDistance = distance
          best = coreLabels[index] ?? NOISE
        }
      })
      return String(best)
    })
}

export function loadDbscanModel(file: unknown): Predict {
  const parsed = dbscanSchema.safeParse(file)
  if (!parsed.success) invalidModel('payload')
  return dbscanPredict({ ...parsed.data, format: DBSCAN_FORMAT })
}
