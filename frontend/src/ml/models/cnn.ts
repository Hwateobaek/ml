/**
 * 합성곱 신경망 모델 형식 (open-decisions.md "합성곱을 직접 학습하는 길을 연다").
 *
 * **이 파일은 TensorFlow.js를 모른다.** 저장된 모델을 다시 여는 일은 그것을 만든
 * 라이브러리와 무관해야 한다는 것이 이 계층의 규칙이고(`models/types.ts`), 합성곱도
 * 예외가 아니다 — **TF.js가 뱉은 조각을 그대로 담지 않고 우리 형식으로 옮겨 담는다.**
 *
 * **예측은 여기서 직접 계산한다.** 순전파뿐이라 할 일이 적다:
 *
 * ```
 * [합성곱 → ReLU → 풀링] × n  →  펴기  →  Dense(ReLU)  →  Dense(softmax)
 * ```
 *
 * 역전파가 없으므로 **기울기도, 최적화기도, 난수도 없다** — 학습 쪽(`engines/tfjs-cnn.ts`)이
 * 지고 있는 무게가 여기엔 하나도 안 온다. 덕분에 `load`가 **동기**로 남고, 예측 화면도
 * 검사도 지금 모양 그대로다.
 *
 * **두 계산이 같은 답을 내는지 쟀다** (2026-10-21, 브라우저). 같은 가중치를 TF.js
 * 그래프에 도로 넣고 확률을 사진마다 대조했을 때 **최대 차이가 1.5 × 10⁻⁷**이었다 —
 * float32의 반올림 폭이고, 구조 셋(기본값 · 커널 4 · 커널 2)에서 모두 그랬다.
 * **짝수 커널을 일부러 넣은 이유**는 `same` 테두리가 위·왼쪽으로 치우치는 자리가
 * 거기뿐이라서다.
 *
 * **사람이 다시 재야 하는 값이다.** 노드에는 TF.js가 없어 검사가 대신 물 수 없고,
 * 그래서 `tests/cnn-model.spec.ts`는 **산수 자체**를 손으로 푼 작은 보기로 못 박는다 —
 * 테두리·ReLU·풀링·펴는 순서·softmax 다섯이다.
 */

import { z } from 'zod'

import { ClientError } from '../../errors'
import type { Prediction } from '../metrics'
import type { ModelFile, Predict, PredictProba, ProbaModel } from './types'

export const CNN_FORMAT = 'mlpx-cnn-v1'

/** 합성곱 한 겹과 그 뒤의 풀링. */
export interface CnnBlock {
  /** 필터 수. 나가는 칸 수다. */
  readonly filters: number
  /** 커널 한 변. */
  readonly kernelSize: number
  /** 이 블록 뒤에 풀링이 있었는가. **없을 수 있다** — 더 접을 자리가 없으면 건너뛴다. */
  readonly pooled: boolean
  /** `[커널높이][커널너비][들어오는 칸][나가는 칸]` 순서로 편 것. base64. */
  readonly kernel: string
  /** 나가는 칸 수만큼. base64. */
  readonly bias: string
}

/** 완전연결 한 겹. */
export interface CnnDense {
  readonly units: number
  /** 마지막 층만 `softmax`다. */
  readonly activation: 'relu' | 'softmax'
  /** `[들어오는 칸][나가는 칸]` 순서로 편 것. base64. */
  readonly kernel: string
  readonly bias: string
}

export interface CnnModel extends ModelFile {
  readonly format: typeof CNN_FORMAT
  /** 출력층의 칸 순서. sklearn의 `classes_`에 해당한다. */
  readonly classes: readonly string[]
  /** 사진 한 변. **이 모델은 이 크기로만 돈다.** */
  readonly inputSize: number
  /** 에포크마다의 훈련 손실. **화면이 이것을 그린다** (`ml/loss-curve.ts`). */
  readonly lossCurve: readonly number[]
  readonly blocks: readonly CnnBlock[]
  readonly dense: readonly CnnDense[]
}

const blockSchema = z.looseObject({
  filters: z.number().int().positive(),
  kernelSize: z.number().int().positive(),
  pooled: z.boolean(),
  kernel: z.string().min(1),
  bias: z.string().min(1),
})

const denseSchema = z.looseObject({
  units: z.number().int().positive(),
  activation: z.union([z.literal('relu'), z.literal('softmax')]),
  kernel: z.string().min(1),
  bias: z.string().min(1),
})

const cnnModelSchema = z.looseObject({
  format: z.literal(CNN_FORMAT),
  classes: z.array(z.string()).min(2),
  inputSize: z.number().int().positive(),
  lossCurve: z.array(z.number()),
  blocks: z.array(blockSchema).min(1),
  dense: z.array(denseSchema).min(1),
})

function invalid(field: string): never {
  throw new ClientError('MODEL_FILE_INVALID', { field })
}

/** 풀링 한 변. **학습 쪽과 같은 값이어야 한다** (`engines/tfjs-cnn.ts`의 `POOL_SIZE`). */
const POOL_SIZE = 2

/** base64 한 덩이를 float32 배열로. **담을 때와 같은 순서다.** */
export function fromBase64(text: string): Float32Array {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let at = 0; at < binary.length; at += 1) bytes[at] = binary.charCodeAt(at)
  // **복사한다.** `bytes.buffer`를 그대로 보면 오프셋이 안 맞는 경우가 생기고,
  // 그때 조용히 다른 숫자를 읽는다.
  return new Float32Array(bytes.buffer.slice(0))
}

/** 검증과 디코딩을 마친 모델. **`load`와 `loadProba`가 같은 것을 본다.** */
export interface ParsedCnn {
  readonly classes: readonly string[]
  readonly inputSize: number
  readonly lossCurve: readonly number[]
  readonly blocks: readonly {
    readonly filters: number
    readonly kernelSize: number
    readonly pooled: boolean
    readonly kernel: Float32Array
    readonly bias: Float32Array
  }[]
  readonly dense: readonly {
    readonly units: number
    readonly activation: 'relu' | 'softmax'
    readonly kernel: Float32Array
    readonly bias: Float32Array
  }[]
}

/**
 * 파일 내용을 확인하고 편다. **안 맞으면 던진다** — 깨진 모델로 그럴듯한 답을 내느니
 * 아무 답도 안 내는 쪽이다.
 *
 * **길이까지 센다.** 스키마는 "글자가 있다"까지만 보는데, 가중치는 **개수가 맞아야**
 * 뜻이 있다 — 한 칸만 모자라도 나머지가 통째로 밀려 조용히 다른 모델이 된다.
 */
export function parseCnn(file: unknown): ParsedCnn {
  const parsed = cnnModelSchema.safeParse(file)
  if (!parsed.success) invalid(parsed.error.issues[0]?.path.join('.') ?? 'format')
  const model = parsed.data as CnnModel

  let channels = 3
  const blocks = model.blocks.map((block, index) => {
    const kernel = fromBase64(block.kernel)
    const bias = fromBase64(block.bias)
    const want = block.kernelSize * block.kernelSize * channels * block.filters
    if (kernel.length !== want) invalid(`blocks.${index}.kernel`)
    if (bias.length !== block.filters) invalid(`blocks.${index}.bias`)
    channels = block.filters
    return { ...block, kernel, bias }
  })

  /** 펴고 나면 칸이 몇 개인가. **블록마다 풀링이 반으로 줄인다.** */
  let side = model.inputSize
  for (const block of model.blocks) {
    if (block.pooled) side = Math.floor(side / POOL_SIZE)
  }
  let incoming = side * side * channels

  const dense = model.dense.map((layer, index) => {
    const kernel = fromBase64(layer.kernel)
    const bias = fromBase64(layer.bias)
    if (kernel.length !== incoming * layer.units) invalid(`dense.${index}.kernel`)
    if (bias.length !== layer.units) invalid(`dense.${index}.bias`)
    incoming = layer.units
    return { ...layer, kernel, bias }
  })

  // **마지막 층의 칸 수가 범주 수와 같아야 한다.** 아니면 어느 칸이 어느 범주인지
  // 아무도 모른다.
  if (dense[dense.length - 1]?.units !== model.classes.length) invalid('dense')

  return {
    classes: model.classes,
    inputSize: model.inputSize,
    lossCurve: model.lossCurve,
    blocks,
    dense,
  }
}

/**
 * 합성곱 한 겹. **`same` 테두리에 보폭 1이라 크기가 안 바뀐다** — 줄이는 일은 풀링이
 * 맡는다 (`engines/tfjs-cnn.ts`의 `padding: 'same'`).
 *
 * `same` 테두리의 자리는 텐서플로가 정한 그대로다: 폭이 `k - 1`이고 **위·왼쪽이
 * 내림**이다. 홀수 커널에서는 양쪽이 같고, 짝수에서 갈린다.
 */
function convolve(
  input: Float32Array,
  side: number,
  inChannels: number,
  kernel: Float32Array,
  bias: Float32Array,
  kernelSize: number,
  filters: number,
): Float32Array {
  const pad = Math.floor((kernelSize - 1) / 2)
  const out = new Float32Array(side * side * filters)
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      for (let f = 0; f < filters; f += 1) {
        let sum = bias[f]!
        for (let ky = 0; ky < kernelSize; ky += 1) {
          const sy = y + ky - pad
          if (sy < 0 || sy >= side) continue
          for (let kx = 0; kx < kernelSize; kx += 1) {
            const sx = x + kx - pad
            if (sx < 0 || sx >= side) continue
            const at = (sy * side + sx) * inChannels
            const weightAt = (ky * kernelSize + kx) * inChannels * filters + f
            for (let c = 0; c < inChannels; c += 1) {
              sum += input[at + c]! * kernel[weightAt + c * filters]!
            }
          }
        }
        // **ReLU를 여기서 함께 한다.** 합성곱 뒤에 언제나 붙으므로 배열을 한 번 덜 만든다.
        out[(y * side + x) * filters + f] = sum > 0 ? sum : 0
      }
    }
  }
  return out
}

/** 2×2 최댓값 풀링, 보폭 2. **남는 줄은 버린다** (`valid` 테두리). */
function pool(input: Float32Array, side: number, channels: number): Float32Array {
  const half = Math.floor(side / POOL_SIZE)
  const out = new Float32Array(half * half * channels)
  for (let y = 0; y < half; y += 1) {
    for (let x = 0; x < half; x += 1) {
      for (let c = 0; c < channels; c += 1) {
        let best = -Infinity
        for (let dy = 0; dy < POOL_SIZE; dy += 1) {
          for (let dx = 0; dx < POOL_SIZE; dx += 1) {
            const value =
              input[((y * POOL_SIZE + dy) * side + (x * POOL_SIZE + dx)) * channels + c]!
            if (value > best) best = value
          }
        }
        out[(y * half + x) * channels + c] = best
      }
    }
  }
  return out
}

/** 완전연결 한 겹. 가중치는 `[들어오는 칸][나가는 칸]` 순서다. */
function dense(
  input: Float32Array,
  kernel: Float32Array,
  bias: Float32Array,
  units: number,
): Float32Array {
  const out = new Float32Array(units)
  for (let u = 0; u < units; u += 1) out[u] = bias[u]!
  for (let i = 0; i < input.length; i += 1) {
    const value = input[i]!
    if (value === 0) continue
    const row = i * units
    for (let u = 0; u < units; u += 1) out[u] = out[u]! + value * kernel[row + u]!
  }
  return out
}

/**
 * softmax. **가장 큰 값을 빼고 지수를 취한다** — 안 그러면 칸이 조금만 커져도 넘친다.
 */
function softmax(values: Float32Array): Float64Array {
  let top = -Infinity
  for (const value of values) if (value > top) top = value
  const out = new Float64Array(values.length)
  let total = 0
  for (let at = 0; at < values.length; at += 1) {
    const value = Math.exp(values[at]! - top)
    out[at] = value
    total += value
  }
  // 분모가 0이 되는 자리는 없다 — 가장 큰 칸이 언제나 `exp(0) = 1`이다.
  for (let at = 0; at < out.length; at += 1) out[at] = out[at]! / total
  return out
}

/**
 * 사진 한 장의 확률. **들어오는 줄은 화소다** — `64 × 64 × 3`을 `[높이][너비][채널]`
 * 순서로 편 것이고, 값은 이미 `[0, 1]`이다 (`ml/pixel-features.ts`).
 */
function forward(model: ParsedCnn, row: readonly number[]): Float64Array {
  const expected = model.inputSize * model.inputSize * 3
  if (row.length !== expected) {
    throw new ClientError('MODEL_FILE_INVALID', { field: 'inputSize' })
  }

  let current: Float32Array = Float32Array.from(row)
  let side = model.inputSize
  let channels = 3
  for (const block of model.blocks) {
    current = convolve(
      current,
      side,
      channels,
      block.kernel,
      block.bias,
      block.kernelSize,
      block.filters,
    )
    channels = block.filters
    if (block.pooled) {
      current = pool(current, side, channels)
      side = Math.floor(side / POOL_SIZE)
    }
  }

  // **펴는 순서가 `[높이][너비][채널]`이다.** 학습 쪽의 `Flatten`과 같아야 하고,
  // 다르면 가중치가 **엉뚱한 칸에 붙으면서 아무 오류도 안 난다.**
  for (const layer of model.dense) {
    current = dense(current, layer.kernel, layer.bias, layer.units)
    if (layer.activation === 'relu') {
      for (let at = 0; at < current.length; at += 1) {
        if (current[at]! < 0) current[at] = 0
      }
    }
  }
  return softmax(current)
}

/** 가장 큰 칸의 범주. **동점이면 앞선 칸이다** — `classes`의 순서가 기준이다. */
function labelOf(model: ParsedCnn, scores: Float64Array): Prediction {
  let best = 0
  for (let at = 1; at < scores.length; at += 1) {
    if (scores[at]! > scores[best]!) best = at
  }
  return model.classes[best]!
}

/** 블록 하나가 내놓은 그림들. **채널 하나가 그림 한 장이다.** */
export interface CnnFeatureMaps {
  /** 몇 번째 블록인가. 1부터 센다 — 학생이 세는 방식이다. */
  readonly block: number
  /** 한 변. 풀링을 지날 때마다 반으로 준다. */
  readonly side: number
  /** 채널(필터) 수. */
  readonly channels: number
  /** `[높이][너비][채널]` 순서. **펴는 순서와 같다.** */
  readonly values: Float32Array
}

/**
 * 블록마다의 **특성맵** — *"합성곱이 무엇을 보는가"*를 그림으로 말하는 자리
 * (open-decisions.md "필터와 특성맵을 보여준다").
 *
 * **`forward`와 같은 계산을 지난다.** 복사본을 만들지 않는 이유가 그것이다 — 두 벌이면
 * 한쪽만 고쳤을 때 **화면이 모델이 보지 않은 그림을 보여준다.**
 *
 * **완전연결 층은 안 지난다.** 거기부터는 그림이 아니라 숫자 줄이고, 그려 봐야 학생이
 * 읽을 것이 없다.
 */
export function cnnFeatureMaps(model: ParsedCnn, row: readonly number[]): CnnFeatureMaps[] {
  const expected = model.inputSize * model.inputSize * 3
  if (row.length !== expected) {
    throw new ClientError('MODEL_FILE_INVALID', { field: 'inputSize' })
  }

  const maps: CnnFeatureMaps[] = []
  let current: Float32Array = Float32Array.from(row)
  let side = model.inputSize
  let channels = 3
  for (const [index, block] of model.blocks.entries()) {
    current = convolve(
      current,
      side,
      channels,
      block.kernel,
      block.bias,
      block.kernelSize,
      block.filters,
    )
    channels = block.filters
    if (block.pooled) {
      current = pool(current, side, channels)
      side = Math.floor(side / POOL_SIZE)
    }
    maps.push({ block: index + 1, side, channels, values: current })
  }
  return maps
}

/**
 * 첫 합성곱층의 **필터들** — 학생이 눈으로 볼 수 있는 유일한 층이다.
 *
 * **첫 층만 그린다.** 두 번째 층부터는 들어오는 칸이 색이 아니라 **앞 층의 필터**라,
 * 3채널 그림으로 그릴 수가 없다 — 그려도 색은 아무 뜻이 없다.
 *
 * 돌려주는 것은 필터마다 `[높이][너비][RGB]`로 편 값이다. **범위는 안 맞춘다** —
 * 가중치는 음수도 되고, 어떻게 보일지는 그리는 쪽이 정한다 (§8.3).
 */
export function cnnFirstFilters(model: ParsedCnn): {
  readonly size: number
  readonly filters: readonly Float32Array[]
} {
  const first = model.blocks[0]
  if (!first) return { size: 0, filters: [] }
  const { kernelSize, filters: count, kernel } = first
  const out: Float32Array[] = []
  for (let f = 0; f < count; f += 1) {
    const tile = new Float32Array(kernelSize * kernelSize * 3)
    for (let y = 0; y < kernelSize; y += 1) {
      for (let x = 0; x < kernelSize; x += 1) {
        for (let c = 0; c < 3; c += 1) {
          // 담긴 순서가 `[ky][kx][들어오는 칸][나가는 칸]`이다.
          tile[(y * kernelSize + x) * 3 + c] = kernel[((y * kernelSize + x) * 3 + c) * count + f]!
        }
      }
    }
    out.push(tile)
  }
  return { size: kernelSize, filters: out }
}

export function loadCnnModel(file: unknown): Predict {
  const model = parseCnn(file)
  return (features) => features.map((row) => labelOf(model, forward(model, row)))
}

/**
 * 확률. **라벨은 `load`와 같은 규칙으로 정한다** (mlpx-spec.md §5.4) — 둘이 갈리면
 * 화면이 "고양이 60%"라고 적고 답은 "개"라고 적는다.
 */
export function loadCnnProba(file: unknown): ProbaModel {
  const model = parseCnn(file)
  const predict: PredictProba = (features) => features.map((row) => forward(model, row))
  return { classes: model.classes, predict }
}
