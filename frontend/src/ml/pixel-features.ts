/**
 * 정본 사진을 **합성곱이 먹는 숫자 행렬**로 옮긴다
 * (open-decisions.md "합성곱을 직접 학습하는 길을 연다").
 *
 * ```
 * 정본 바이트 → createImageBitmap → 64×64로 줄여 그리기 → RGB를 255로 나누기
 * ```
 *
 * **백본이 지나는 길과 같은 모양이고 목적지가 다르다** (`ml/embed/runner.ts`). 저쪽은
 * 224 그대로를 그래프에 넣고, 여기는 **줄여서** 학생이 쌓은 층에 넣는다. 화소를 옮기는
 * 산수는 둘이 같은 함수를 쓴다 (`ml/embed/pixels.ts`의 `packPixels`).
 *
 * **255로 나누는 것이 교과서의 `rescale = 1./255`다.** 전처리 화면의 스케일링 선택지와
 * 다른 물건이라 거기 안 올린다 — 그쪽은 임베딩의 축을 다루고 이것은 **모델 안에 박힌
 * 상수**다.
 *
 * **그리는 일은 이 파일 밖에 있다.** `OffscreenCanvas`도 `createImageBitmap`도 워커
 * 전역이라, 넣어 두면 이 계층이 노드에서 못 돈다 — `ml/embed/runner.ts`가 인터페이스를
 * 따로 둔 것과 같은 이유다. 여기 있는 것은 **계약과 산수**뿐이다.
 */

import { CNN_INPUT_SIZE } from '../limits'
import { packPixels } from './embed/pixels'

/** 화소를 `[0, 1]`로 옮긴다. 교과서의 `rescale = 1./255`. */
const PIXEL_RANGE: readonly [number, number] = [0, 1]

/** 사진 한 장이 차지하는 칸 수. `64 × 64 × 3`. */
export const PIXEL_FEATURE_COUNT = CNN_INPUT_SIZE * CNN_INPUT_SIZE * 3

/**
 * 정본 바이트를 화소 행렬로. **한 줄이 사진 한 장이다** — 표의 한 행과 같은 자리라
 * 그 뒤는 지금 있는 것이 그대로 돈다 (`ml/experiment.ts`의 `trainOne`).
 *
 * **비동기인 것은 디코드 때문이다.** `createImageBitmap`이 약속을 돌려준다.
 */
export type PixelDecoder = (images: readonly Uint8Array[]) => Promise<number[][]>

/**
 * 그리는 수단. **워커가 준다** — 검사는 가짜를 준다.
 *
 * `size × size`로 **줄여 그린 뒤** RGBA 바이트를 돌려준다.
 */
export type DrawToRgba = (bytes: Uint8Array, size: number) => Promise<Uint8ClampedArray>

/**
 * 그리는 수단 하나로 디코더를 만든다.
 *
 * **한 장씩 돈다.** 전부를 한꺼번에 비트맵으로 열면 사진 수만큼 비트맵이 동시에 살고,
 * 그것이 정확히 저사양 교실 PC가 먼저 무너지는 자리다 — 백본 러너도 한 장씩 돈다.
 */
export function pixelDecoder(draw: DrawToRgba, size: number = CNN_INPUT_SIZE): PixelDecoder {
  const perImage = size * size * 3
  return async (images) => {
    const rows: number[][] = []
    const scratch = new Float32Array(perImage)
    for (const bytes of images) {
      const rgba = await draw(bytes, size)
      packPixels(rgba, size, PIXEL_RANGE, scratch)
      // **옮겨 담는다.** `scratch`를 그대로 담으면 모든 줄이 같은 배열을 가리킨다.
      rows.push(Array.from(scratch))
    }
    return rows
  }
}
