/**
 * 합성곱 신경망을 **직접 학습한다** (open-decisions.md "합성곱을 직접 학습하는 길을 연다").
 *
 * ```
 * [합성곱 → 풀링] × n  →  Flatten  →  Dense(relu)  →  Dense(범주 수, softmax)
 * ```
 *
 * **교과서 151쪽의 모양이다.** 거기서 `model.add(vgg)`였던 첫 칸이 **학생이 쌓은
 * 합성곱층**으로 바뀐 것뿐이고, 손실(`categorical_crossentropy`)도 최적화(`adam`)도
 * 그대로다.
 *
 * **등록부에서 유일하게 화소를 받는다** (`ml/algorithms.ts`의 `featureSource`). 들어오는
 * `features`의 한 줄이 사진 한 장이고 길이가 `64 × 64 × 3`이다 — 그것을 여기서
 * `[장수, 64, 64, 3]`으로 접는다.
 *
 * **TF.js는 지연 로딩한다.** 합성곱을 안 고른 학생이 250KB를 치르면 안 된다 —
 * 백본 러너(`ml/embed/runner.ts`)가 `tfjs-core`를 그렇게 쓰는 것과 같은 자리다.
 *
 * **씨앗을 반드시 쓴다.** 가중치 초기화에 `randomState`를 넣고, 배치 순서는 우리
 * 셔플(`ml/shuffle.ts`)로 미리 섞어 `shuffle: false`로 넘긴다 — TF.js의 `shuffle: true`는
 * 씨앗을 안 받아서, 쓰면 **같은 설정으로 두 번 돌릴 때 곡선이 달라진다.**
 */

import { CNN_INPUT_SIZE } from '../../limits'
import type { Prediction } from '../metrics'
import { CNN_FORMAT, type CnnModel } from '../models/cnn'
import type { Predict } from '../models/types'
import { shuffled } from '../shuffle'
import type { FitInput, FitResult } from './mljs'
import { TFJS_PARAMETERS } from './tfjs-params'
import type { HyperparameterSpec } from '../hyperparams'

export const TFJS_ALGORITHMS: readonly string[] = ['cnn']

/**
 * 엔진 이름과 판. **`run.engine`에 그대로 들어간다** — 재실행 대조가 이 값으로
 * 엔진을 가린다. 올리는 일은 지시가 있을 때만이다 (CLAUDE.md §4).
 */
export const TFJS_ENGINE = { kind: 'tfjs', version: '1' } as const

/** 완전연결층의 활성. 교과서가 `relu`다. */
const DENSE_ACTIVATION = 'relu'

/** 풀링 한 변. 2면 화면이 반으로 준다 — 교과서의 VGG16도 2다. */
const POOL_SIZE = 2

export function parameters(algorithm: string): readonly HyperparameterSpec[] {
  return TFJS_PARAMETERS[algorithm] ?? []
}

/**
 * 확정된 값에서 수치를 읽는다. **`resolve()`가 이미 채웠다** — 0을 준비해 두는 것은
 * 타입 때문이지 기본값이 아니다 (`mljs.ts`의 `numberOption`과 같은 자리).
 */
function numberOption(source: Record<string, unknown>, name: string): number {
  const value = source[name]
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

export function resolve(
  algorithm: string,
  given: Record<string, unknown>,
): Record<string, unknown> {
  const specs = parameters(algorithm)
  const resolved: Record<string, unknown> = { ...given }
  for (const spec of specs) {
    const value = given[spec.name]
    if (typeof value === 'number' && Number.isFinite(value)) {
      resolved[spec.name] = spec.integer ? Math.round(value) : value
    } else {
      resolved[spec.name] = spec.default
    }
  }
  return resolved
}

/** 범주 이름을 고정된 순서로. **파일에 적히는 `classes`가 이것이다.** */
function classesOf(target: readonly Prediction[]): string[] {
  return [...new Set(target.map(String))].sort()
}

/**
 * 층이 몇 번 반으로 접히고도 남는가.
 *
 * **풀링이 1×1 아래로 못 내려간다.** 학생이 합성곱층 넷을 고르고 입력이 작으면 그런
 * 자리가 생기는데, TF.js는 거기서 모양 오류로 던진다 — **학생이 읽을 수 없는 문장이다.**
 * 그래서 **접을 수 있는 만큼만 접는다**: 줄일 자리가 없으면 풀링을 건너뛰고 합성곱만
 * 쌓는다.
 */
export function poolableBlocks(size: number, blocks: number): number {
  let remaining = size
  let count = 0
  while (count < blocks && remaining >= POOL_SIZE * 2) {
    remaining = Math.floor(remaining / POOL_SIZE)
    count += 1
  }
  return count
}

/** TF.js 모듈 묶음. **워커 안에서 한 번 받아 둔다.** */
type TfLayers = typeof import('@tensorflow/tfjs-layers')
type TfCore = typeof import('@tensorflow/tfjs-core')

let loaded: { core: TfCore; layers: TfLayers } | null = null

/**
 * TF.js를 받아 둔다. **백본 러너와 같은 순서로 백엔드를 고른다** — WebGPU → WebGL →
 * wasm이고, 되는 것으로 돈다.
 */
async function load(): Promise<{ core: TfCore; layers: TfLayers }> {
  if (loaded) return loaded
  const core = await import('@tensorflow/tfjs-core')
  try {
    await import('@tensorflow/tfjs-backend-webgl')
    await core.setBackend('webgl')
  } catch {
    // 백엔드가 없으면 TF.js의 기본(cpu)으로 돈다. 느리지만 답은 같다.
  }
  await core.ready()
  const layers = await import('@tensorflow/tfjs-layers')
  loaded = { core, layers }
  return loaded
}

/**
 * 학습한다.
 *
 * **여기서 던지는 것은 전부 실패한 run 하나가 된다** (`ml/experiment.ts`의 `trainOne`).
 * 실험은 계속 돈다.
 */
export async function fit(algorithm: string, input: FitInput): Promise<FitResult> {
  if (algorithm !== 'cnn') throw new Error(`unknown algorithm: ${algorithm}`)

  const { core, layers } = await load()
  const options = resolve(algorithm, input.hyperparameters)
  const size = CNN_INPUT_SIZE
  const perImage = size * size * 3

  const classes = classesOf(input.target)
  const rows = input.features.length
  if (rows === 0) throw new Error('no training photos')
  if (input.features[0]?.length !== perImage) {
    throw new Error(`expected ${perImage} pixel features, got ${input.features[0]?.length ?? 0}`)
  }

  /**
   * **순서를 우리가 섞는다.** TF.js의 `shuffle: true`는 씨앗을 안 받는다 —
   * 그대로 두면 재현이 깨진다 (이 파일의 머리말).
   */
  const order = shuffled(
    Array.from({ length: rows }, (_unused, index) => index),
    input.randomState,
  )

  const flat = new Float32Array(rows * perImage)
  const labels = new Float32Array(rows * classes.length)
  for (const [seat, source] of order.entries()) {
    const row = input.features[source]!
    flat.set(row as number[], seat * perImage)
    const slot = classes.indexOf(String(input.target[source]))
    if (slot >= 0) labels[seat * classes.length + slot] = 1
  }

  const blocks = Math.max(1, numberOption(options, 'convBlocks'))
  const pooled = poolableBlocks(size, blocks)
  const model = layers.sequential()
  /**
   * 쌓은 층을 그대로 들고 있는다. **파일로 옮겨 담을 때 이 목록을 읽는다**
   * (아래 `serialize`) — TF.js가 붙인 이름으로 되찾으면 그 이름이 판마다 달라질 수 있고,
   * 그때 **엉뚱한 층의 가중치를 담으면서 아무 오류도 안 난다.**
   */
  const convShapes: { filters: number; kernelSize: number; pooled: boolean }[] = []
  const denseShapes: { units: number; activation: 'relu' | 'softmax' }[] = []
  for (let block = 0; block < blocks; block += 1) {
    const filters = Math.max(1, numberOption(options, 'filters')) * 2 ** block
    const kernelSize = Math.max(1, numberOption(options, 'kernelSize'))
    convShapes.push({ filters, kernelSize, pooled: block < pooled })
    model.add(
      layers.layers.conv2d({
        // **블록마다 두 배다.** 교과서가 보여 주는 VGG16의 모양이기도 하다(64 → 128 → 256).
        filters,
        kernelSize,
        activation: 'relu',
        // **`same`이라 합성곱이 크기를 안 깎는다.** 줄이는 일은 풀링 하나가 맡는다 —
        // 그래야 "풀링이 화면을 반으로 줄인다"는 수업 문장이 화면에서 참이 된다.
        padding: 'same',
        kernelInitializer: layers.initializers.glorotUniform({ seed: input.randomState + block }),
        ...(block === 0 ? { inputShape: [size, size, 3] } : {}),
      }),
    )
    if (block < pooled) model.add(layers.layers.maxPooling2d({ poolSize: POOL_SIZE }))
  }
  model.add(layers.layers.flatten())
  const denseUnits = Math.max(1, numberOption(options, 'denseUnits'))
  denseShapes.push({ units: denseUnits, activation: DENSE_ACTIVATION })
  model.add(
    layers.layers.dense({
      units: denseUnits,
      activation: DENSE_ACTIVATION,
      kernelInitializer: layers.initializers.glorotUniform({ seed: input.randomState }),
    }),
  )
  denseShapes.push({ units: classes.length, activation: 'softmax' })
  model.add(
    layers.layers.dense({
      units: classes.length,
      activation: 'softmax',
      kernelInitializer: layers.initializers.glorotUniform({ seed: input.randomState + 1 }),
    }),
  )
  model.compile({
    // 교과서 그대로다 (`loss='categorical_crossentropy', optimizer='adam'`).
    loss: 'categoricalCrossentropy',
    optimizer: core.train.adam(numberOption(options, 'learningRate')),
    metrics: ['accuracy'],
  })

  const lossCurve: number[] = []
  const xs = core.tensor4d(flat, [rows, size, size, 3])
  const ys = core.tensor2d(labels, [rows, classes.length])
  try {
    await model.fit(xs, ys, {
      epochs: Math.max(1, numberOption(options, 'epochs')),
      batchSize: Math.min(Math.max(1, numberOption(options, 'batchSize')), rows),
      shuffle: false,
      callbacks: {
        onEpochEnd: (_epoch, logs) => {
          // **엔진이 실제로 지나간 손실을 그대로 담는다** (`ml/loss-curve.ts`) — 평활하지도
          // 로그를 씌우지도 않는다.
          const loss = logs?.['loss']
          if (typeof loss === 'number' && Number.isFinite(loss)) lossCurve.push(loss)
        },
      },
    })
  } finally {
    // **들어온 텐서를 반드시 놓는다.** 사진 800장이 한 덩이에 39MB이고, 이것이
    // 안 놓이면 학생이 설정을 바꿔 다시 돌릴 때마다 그만큼이 GPU에 쌓인다.
    xs.dispose()
    ys.dispose()
  }

  /**
   * 예측. **한 번에 다 넣지 않는다** — 예측 몫이 클 때 텐서 하나가 그만큼 커진다.
   *
   * **동기다.** 모델 계층의 `Predict`가 동기라서인데, TF.js의 `predict`도 동기로
   * 답을 주므로(`dataSync`) 여기서는 맞출 수 있다.
   */
  const predict: Predict = (features) => {
    if (features.length === 0) return []
    const batch = new Float32Array(features.length * perImage)
    for (const [index, row] of features.entries()) batch.set(row as number[], index * perImage)
    const input4d = core.tensor4d(batch, [features.length, size, size, 3])
    try {
      const output = model.predict(input4d)
      const scores = (Array.isArray(output) ? output[0]! : output).dataSync()
      return features.map((_unused, index) => {
        let best = 0
        for (let slot = 1; slot < classes.length; slot += 1) {
          if (scores[index * classes.length + slot]! > scores[index * classes.length + best]!) {
            best = slot
          }
        }
        return classes[best]!
      })
    } finally {
      input4d.dispose()
    }
  }

  /**
   * **우리 형식으로 옮겨 담는다** (`ml/models/cnn.ts`). TF.js가 뱉은 조각을 그대로
   * 담지 않는 이유는, 그러면 **그 파일을 여는 쪽이 TF.js를 알아야 하기** 때문이다 —
   * 모델 계층이 금지하는 바로 그것이다 (`ml/models/types.ts`).
   *
   * **못 담아도 학습은 성공이다** (mlpx-spec.md 4.2). 그때는 사유만 적히고 지표는 남는다.
   */
  let file: CnnModel | undefined
  let omitted: string | undefined
  try {
    file = serialize(model, convShapes, denseShapes, classes, size, lossCurve)
  } catch (error) {
    omitted = error instanceof Error ? error.message : String(error)
  }

  return {
    predict,
    ...(file ? { model: file } : {}),
    ...(omitted === undefined ? {} : { modelOmittedDetail: omitted }),
  }
}

/**
 * float32 배열을 base64로. **`.mlpx` 안이 JSON이라 글자여야 한다.**
 *
 * 조각으로 끊어 읽는 이유는 `String.fromCharCode`가 **인자 수 상한**에 걸리기 때문이다 —
 * 가중치 50만 개면 바이트가 200만 개다.
 */
function toBase64(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength)
  let text = ''
  const CHUNK = 0x8000
  for (let at = 0; at < bytes.length; at += CHUNK) {
    text += String.fromCharCode(...bytes.subarray(at, at + CHUNK))
  }
  return btoa(text)
}

/**
 * 층 하나의 `[커널, 절편]`. **TF.js가 그 순서로 준다.**
 *
 * **`dataSync()`가 float32인지 확인한다.** 타입은 세 가지를 다 열어 두는데, 우리가 쌓은
 * 층의 가중치는 언제나 float32다 — 아니면 **바이트를 잘못 읽으면서 조용히 다른 숫자가
 * 된다.**
 */
function weightsOf(layer: import('@tensorflow/tfjs-layers').layers.Layer): {
  kernel: string
  bias: string
} {
  const [kernel, bias] = layer.getWeights()
  if (!kernel || !bias) throw new Error('layer has no weights')
  const floats = (tensor: (typeof kernel)['dataSync'] extends never ? never : typeof kernel) => {
    const values = tensor.dataSync()
    if (!(values instanceof Float32Array)) throw new Error(`expected float32, got ${tensor.dtype}`)
    return values
  }
  return { kernel: toBase64(floats(kernel)), bias: toBase64(floats(bias)) }
}

function serialize(
  model: import('@tensorflow/tfjs-layers').Sequential,
  convShapes: readonly { filters: number; kernelSize: number; pooled: boolean }[],
  denseShapes: readonly { units: number; activation: 'relu' | 'softmax' }[],
  classes: readonly string[],
  inputSize: number,
  lossCurve: readonly number[],
): CnnModel {
  /**
   * **가중치를 든 층만 센다.** 풀링과 펴기는 배울 것이 없어 `getWeights()`가 비어 있고,
   * 그래서 순서가 합성곱들 다음에 완전연결들이다 — 쌓은 차례 그대로다.
   */
  const weighted = model.layers.filter((layer) => layer.getWeights().length > 0)
  if (weighted.length !== convShapes.length + denseShapes.length) {
    throw new Error(`expected ${convShapes.length + denseShapes.length} weighted layers`)
  }

  return {
    format: CNN_FORMAT,
    classes: [...classes],
    inputSize,
    lossCurve: [...lossCurve],
    blocks: convShapes.map((shape, index) => ({ ...shape, ...weightsOf(weighted[index]!) })),
    dense: denseShapes.map((shape, index) => ({
      ...shape,
      ...weightsOf(weighted[convShapes.length + index]!),
    })),
  }
}
