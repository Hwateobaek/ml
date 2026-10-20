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
  for (let block = 0; block < blocks; block += 1) {
    model.add(
      layers.layers.conv2d({
        // **블록마다 두 배다.** 교과서가 보여 주는 VGG16의 모양이기도 하다(64 → 128 → 256).
        filters: Math.max(1, numberOption(options, 'filters')) * 2 ** block,
        kernelSize: Math.max(1, numberOption(options, 'kernelSize')),
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
  model.add(
    layers.layers.dense({
      units: Math.max(1, numberOption(options, 'denseUnits')),
      activation: DENSE_ACTIVATION,
      kernelInitializer: layers.initializers.glorotUniform({ seed: input.randomState }),
    }),
  )
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

  const xs = core.tensor4d(flat, [rows, size, size, 3])
  const ys = core.tensor2d(labels, [rows, classes.length])
  try {
    await model.fit(xs, ys, {
      epochs: Math.max(1, numberOption(options, 'epochs')),
      batchSize: Math.min(Math.max(1, numberOption(options, 'batchSize')), rows),
      shuffle: false,
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
   * **모델을 아직 파일에 안 담는다** (open-decisions.md "합성곱을 직접 학습하는 길을
   * 연다"). 담을 수는 있는데 **다시 읽을 길이 없다** — 모델 계층의 해석기는
   * `load(file): Predict`가 **동기**이고, TF.js 그래프를 세우는 일은 비동기다.
   * 해석기 없는 모델을 넣으면 학생은 **열어서 예측할 수 없는 무게**만 얻는다
   * (`project/attach.ts`가 그래서 떨어뜨린다).
   *
   * 그래서 이 실행은 **지표까지만 남는다** — 파일에는 `modelOmitted:
   * 'engineUnsupported'`가 적히고, 그 어휘가 뜻하는 바가 정확히 이 상태다
   * (`project/schema.ts`: *"그 실행 방법에 아직 직렬화기가 없다"*).
   *
   * **손실 곡선도 그래서 아직 없다.** 곡선은 모델 파일 안에 사는데
   * (`ml/loss-curve.ts`) 그 파일이 안 담기기 때문이다 — 해석기 계층이 비동기를
   * 받는 날 **모델·곡선·예측 셋이 함께** 선다.
   */
  return { predict }
}
