/**
 * 합성곱이 **무엇을 보는가**를 화면에 넘기는 자리
 * (open-decisions.md "필터와 특성맵을 보여준다").
 *
 * **어느 형식이 필터를 갖는지는 이 파일이 안다. 화면은 모른다** (§9.1) —
 * `ml/loss-curve.ts`와 같은 자리이고 같은 이유다.
 *
 * **여기서 그림을 만들지 않는다.** 모델이 들고 있는 값을 꺼내 줄 뿐이고, 그것을
 * 어떻게 보일지(범위를 어디에 맞출지, 몇 장을 그릴지)는 화면이 정한다 (§8.3).
 */

import { CNN_FORMAT, cnnFeatureMaps, cnnFirstFilters, parseCnn, type ParsedCnn } from './models'

export type { CnnFeatureMaps } from './models/cnn'

/**
 * 이 형식이 필터를 갖는가. **형식으로 판정한다. 알고리즘 이름이 아니다** —
 * 같은 알고리즘이 다른 형식으로 담길 수 있고, 우리가 읽을 수 있는지는 형식이 답한다.
 */
export function showsFilters(format: string | undefined): boolean {
  return format === CNN_FORMAT
}

/**
 * 모델을 펴서 돌려준다. **하나라도 안 맞으면 `null`이고, 그때 화면은 아무것도 안 그린다**
 * (§9.2 "없는 것을 이름으로 말하지 않는다").
 *
 * `null`이 되는 자리는 셋이다 — 필터를 안 갖는 형식 · 모델이 파일에 안 담긴 실행
 * (`run.modelOmitted`) · 남이 편집해 깨진 파일.
 */
export function cnnModelOf(
  format: string | undefined,
  bytes: Uint8Array | undefined,
): ParsedCnn | null {
  if (!showsFilters(format) || !bytes) return null
  try {
    return parseCnn(JSON.parse(new TextDecoder().decode(bytes)) as unknown)
  } catch {
    // 못 읽는 파일이다. 이유를 말하는 자리는 다른 곳이고, 여기서는 안 그린다.
    return null
  }
}

export { cnnFeatureMaps, cnnFirstFilters }
