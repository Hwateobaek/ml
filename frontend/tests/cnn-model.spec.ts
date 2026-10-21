/**
 * 합성곱 모델의 **순전파 산수** (`ml/models/cnn.ts`).
 *
 * **왜 손으로 푼 보기인가.** 학습은 TF.js가 하고 예측은 우리가 한다. 두 계산이 같은
 * 답을 내야 하는데, **노드에는 TF.js가 없어 대조할 상대가 없다** — 브라우저에서 사람이
 * 한 번 대조하고(2026-10-21), 여기서는 **산수 자체**를 못 박는다.
 *
 * 그래서 보기는 작고 전부 손으로 풀 수 있다. 커널 하나, 화소 넷 같은 크기다.
 *
 * **여기가 지키는 것은 다섯이다** — `same` 테두리의 자리, ReLU, 풀링, 펴는 순서,
 * softmax. 그중 **펴는 순서**가 가장 조용히 틀리는 자리다: 어긋나도 예외가 안 나고
 * 성적만 나빠진다.
 */

import { describe, expect, it } from 'vitest'

import { CNN_FORMAT, loadCnnModel, loadCnnProba, parseCnn } from '../src/ml/models'

/** float32 배열을 모델 파일이 담는 모양(base64)으로. **엔진이 담는 것과 같은 순서다.** */
function b64(values: readonly number[]): string {
  const floats = Float32Array.from(values)
  const bytes = new Uint8Array(floats.buffer)
  let text = ''
  for (const byte of bytes) text += String.fromCharCode(byte)
  return btoa(text)
}

/**
 * 가장 작은 모델. 2×2 사진, 커널 1×1, 필터 하나, 풀링 없음, 출력 둘.
 *
 * 커널이 `[1, 0, 0]`이라 **빨강만 본다**. 그래서 합성곱의 결과가 곧 빨강 값이다.
 */
function tinyModel(options: {
  convBias?: number
  denseKernel?: readonly number[]
  denseBias?: readonly number[]
}): Record<string, unknown> {
  return {
    format: CNN_FORMAT,
    classes: ['a', 'b'],
    inputSize: 2,
    lossCurve: [1, 0.5],
    blocks: [
      {
        filters: 1,
        kernelSize: 1,
        pooled: false,
        // [ky][kx][채널][필터] = [1][1][3][1]
        kernel: b64([1, 0, 0]),
        bias: b64([options.convBias ?? 0]),
      },
    ],
    dense: [
      {
        units: 2,
        activation: 'softmax',
        // 펴고 나면 2×2×1 = 네 칸. [들어오는 칸][나가는 칸] = [4][2]
        kernel: b64(options.denseKernel ?? [1, 0, 0, 0, 0, 0, 0, 0]),
        bias: b64(options.denseBias ?? [0, 0]),
      },
    ],
  }
}

/** 2×2 사진 하나를 `[높이][너비][채널]` 순서로 편다. */
function photo(reds: readonly [number, number, number, number]): number[] {
  return reds.flatMap((red) => [red, 0, 0])
}

describe('합성곱 모델을 읽는다', () => {
  it('모양이 안 맞으면 던진다 - 한 칸만 모자라도 나머지가 통째로 밀린다', () => {
    const blocks = tinyModel({}).blocks as Record<string, unknown>[]
    // 커널을 두 칸짜리로 바꾼다. 필요한 것은 `1 × 1 × 3 × 1 = 3`칸이다.
    const broken = { ...tinyModel({}), blocks: [{ ...blocks[0]!, kernel: b64([1, 0]) }] }
    expect(() => parseCnn(broken)).toThrow()
  })

  it('마지막 층의 칸 수가 범주 수와 다르면 던진다', () => {
    const broken = { ...tinyModel({}), classes: ['a', 'b', 'c'] }
    expect(() => parseCnn(broken)).toThrow()
  })

  it('손실 곡선을 그대로 돌려준다 - 손대지 않는다', () => {
    expect([...parseCnn(tinyModel({})).lossCurve]).toEqual([1, 0.5])
  })
})

describe('순전파가 손으로 푼 값과 같다', () => {
  /**
   * **펴는 순서가 `[높이][너비][채널]`이다.**
   *
   * 커널이 빨강만 보므로 합성곱 결과는 네 화소의 빨강 값 그대로다. 완전연결의 커널을
   * `[1,0, 0,0, 0,0, 0,0]`으로 두면 **첫 칸만** `a`로 간다 — 그 첫 칸이 왼쪽 위
   * 화소여야 한다.
   */
  it('왼쪽 위 화소가 펴진 첫 칸이다', () => {
    const proba = loadCnnProba(tinyModel({}))
    // 왼쪽 위만 밝다 → a 쪽 점수가 1, b는 0 → softmax로 a가 크다.
    const bright = proba.predict([photo([1, 0, 0, 0])])[0]!
    expect(bright[0]).toBeGreaterThan(bright[1]!)
    // 오른쪽 위만 밝다 → 첫 칸은 0이라 둘 다 0 → 반반이다.
    const other = proba.predict([photo([0, 1, 0, 0])])[0]!
    expect(other[0]).toBeCloseTo(0.5, 10)
    expect(other[1]).toBeCloseTo(0.5, 10)
  })

  it('softmax가 실제 값과 같다', () => {
    const proba = loadCnnProba(tinyModel({}))
    const scores = proba.predict([photo([2, 0, 0, 0])])[0]!
    // 점수는 [2, 0]이다. softmax = [e²/(e²+1), 1/(e²+1)]
    const top = Math.exp(2) / (Math.exp(2) + 1)
    expect(scores[0]).toBeCloseTo(top, 10)
    expect(scores[1]).toBeCloseTo(1 - top, 10)
    expect(scores[0]! + scores[1]!).toBeCloseTo(1, 10)
  })

  it('ReLU가 음수를 0으로 만든다 - 합성곱 뒤에서', () => {
    // 절편을 -1로 두면 빨강 0.5인 화소의 합성곱 결과가 -0.5이고, ReLU가 0으로 만든다.
    const proba = loadCnnProba(tinyModel({ convBias: -1 }))
    const scores = proba.predict([photo([0.5, 0, 0, 0])])[0]!
    // 네 칸이 전부 0 → 점수 [0, 0] → 반반.
    expect(scores[0]).toBeCloseTo(0.5, 10)
  })

  it('절편이 더해진다', () => {
    const proba = loadCnnProba(tinyModel({ denseBias: [1, 0] }))
    // 사진이 전부 0이라 점수는 절편뿐이다: [1, 0].
    const scores = proba.predict([photo([0, 0, 0, 0])])[0]!
    const top = Math.exp(1) / (Math.exp(1) + 1)
    expect(scores[0]).toBeCloseTo(top, 10)
  })

  it('가장 큰 칸의 범주를 답한다 - 확률과 같은 답이다', () => {
    const file = tinyModel({ denseKernel: [0, 1, 0, 0, 0, 0, 0, 0] })
    const predict = loadCnnModel(file)
    const proba = loadCnnProba(file)
    const row = photo([3, 0, 0, 0])
    expect(predict([row])).toEqual(['b'])
    const scores = proba.predict([row])[0]!
    expect(scores[1]).toBeGreaterThan(scores[0]!)
  })

  it('화소 수가 안 맞으면 던진다 - 임베딩을 사진으로 알고 넣는 것을 막는다', () => {
    const predict = loadCnnModel(tinyModel({}))
    expect(() => predict([[1, 2, 3]])).toThrow()
  })
})

describe('풀링과 same 테두리', () => {
  /**
   * 4×4 사진에 3×3 커널 하나, 풀링 한 번. **커널이 전부 1이라 합성곱이 이웃의 합이다.**
   *
   * `same` 테두리라 결과도 4×4이고, 풀링이 2×2로 줄인다. 왼쪽 위 2×2 칸의 최댓값이
   * 펴진 첫 칸이 된다.
   */
  function boxModel(): Record<string, unknown> {
    return {
      format: CNN_FORMAT,
      classes: ['a', 'b'],
      inputSize: 4,
      lossCurve: [],
      blocks: [
        {
          filters: 1,
          kernelSize: 3,
          pooled: true,
          // 3×3×3×1. 빨강만 보고 나머지 채널은 0이다.
          kernel: b64(Array.from({ length: 27 }, (_unused, at) => (at % 3 === 0 ? 1 : 0))),
          bias: b64([0]),
        },
      ],
      // 펴면 2×2×1 = 네 칸.
      dense: [
        {
          units: 2,
          activation: 'softmax',
          kernel: b64([1, 0, 0, 0, 0, 0, 0, 0]),
          bias: b64([0, 0]),
        },
      ],
    }
  }

  it('same 테두리가 크기를 안 깎고 풀링이 반으로 줄인다', () => {
    const proba = loadCnnProba(boxModel())
    // 왼쪽 위 화소만 1. 3×3 합이 닿는 자리는 (0,0)(0,1)(1,0)(1,1) 넷이고 전부 1이다.
    // 풀링의 왼쪽 위 2×2 최댓값도 1 → 첫 칸이 1 → 점수 [1, 0].
    const scores = proba.predict([
      [1, 0, 0, ...Array.from({ length: 45 }, () => 0)] as number[],
    ])[0]!
    const top = Math.exp(1) / (Math.exp(1) + 1)
    expect(scores[0]).toBeCloseTo(top, 10)
  })

  it('먼 구석의 화소는 첫 칸에 안 닿는다', () => {
    const proba = loadCnnProba(boxModel())
    // 오른쪽 아래 화소(3,3)만 1. 풀링의 왼쪽 위 칸(원본 0~1행·0~1열)에는 3×3 합이
    // 닿지 않는다 → 첫 칸이 0 → 반반.
    const row = Array.from({ length: 48 }, () => 0)
    row[15 * 3] = 1
    const scores = proba.predict([row])[0]!
    expect(scores[0]).toBeCloseTo(0.5, 10)
  })
})
