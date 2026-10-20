/**
 * 정본을 캔버스에 **줄여 그려** 화소를 꺼낸다 (`ml/pixel-features.ts`의 `DrawToRgba`).
 *
 * **워커 전역이 필요한 쪽이 여기 전부다.** 산수는 저쪽에 있고 여기 있는 것은 그리는
 * 일뿐이라, 노드에서 이 파일을 임포트해도 **부르지만 않으면** 아무 일도 안 난다 —
 * 화소를 받는 알고리즘을 아무도 안 고르면 한 번도 안 불린다.
 *
 * **백본 러너와 다른 점이 하나 있다** (`ml/embed/runner.ts`). 저쪽은 정본이 요구 크기와
 * 다르면 **던진다** — 없는 화소를 만들지 않기 위해서다. 여기는 **줄이는 쪽**이라 그 걱정이
 * 없다: 224 정본을 64로 줄이는 것은 화소를 만드는 일이 아니라 버리는 일이다.
 */

import { pixelDecoder, type PixelDecoder } from '../pixel-features'

/**
 * 흰 바탕에 줄여 그린다.
 *
 * **바탕을 흰색으로 칠한다.** 캔버스의 초기 상태는 **투명 검정**이고, 알파를 버리는
 * `packPixels`는 그 자리를 **검정**으로 읽는다 — 파이프라인 나머지가 전부 "여백은
 * 흰색"으로 서 있다(`data/image/bake.ts`). 여기서만 검정이면 그 사진이 조용히 다른
 * 값이 된다.
 */
async function drawToRgba(bytes: Uint8Array, size: number): Promise<Uint8ClampedArray> {
  const canvas = new OffscreenCanvas(size, size)
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('OffscreenCanvas 2d context unavailable')

  // **형식을 안 적는다.** `createImageBitmap`이 바이트를 스니핑해 형식을 가리고, 정본은
  // webp일 수도 jpg일 수도 있다 (open-decisions.md "정본은 WebP로 굽는다").
  const bitmap = await createImageBitmap(new Blob([bytes as Uint8Array<ArrayBuffer>]))
  try {
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, size, size)
    context.drawImage(bitmap, 0, 0, size, size)
    return context.getImageData(0, 0, size, size).data
  } finally {
    bitmap.close()
  }
}

/** 학습 워커가 실험에 넘기는 디코더. */
export const canvasPixelDecoder: PixelDecoder = pixelDecoder(drawToRgba)
