/**
 * 주성분 분석 (`open-decisions.md` "사진의 특성을 줄여서 배운다").
 *
 * **왜 있는가.** 사진 한 장이 1,280차원이라 트리 계열의 분할 탐색이 열 수에 그대로
 * 붙는다 — 같은 1,000장에서 **50차원으로 줄이면 38배 빨라졌다**(2026-10-19 실측,
 * `limits.ts`). 줄이는 일을 학생이 고르게 하는 것이 이 모듈이고, 수업에서는 *"사진
 * 한 장이 숫자 1,280개인데 그중 50개만 남겨도 되나"*라는 이야깃거리가 된다.
 *
 * **무작위 SVD다** (Halko et al. 2011). sklearn의 `PCA(svd_solver='randomized')`와 같은
 * 계통이고, 그쪽도 큰 행렬에서는 이것을 기본으로 고른다. 전체 SVD를 피하는 이유는
 * 비용이다 — 1,280×1,280을 통째로 분해하면 교실 PC에서 몇 분이 걸린다.
 *
 * **sklearn과 숫자가 같지는 않다.** 난수가 다르므로 성분의 부호와 끝자리가 갈릴 수
 * 있다. 대조 픽스처는 없고, 이 파일이 지키는 것은 **성질**이다(직교성·분산 순서·
 * 되살림 오차·같은 씨앗이면 같은 결과). `tests/pca.spec.ts`가 그 넷을 본다.
 *
 * **남의 수치 라이브러리를 안 쓴다.** `ml-matrix`로 쓰면 다섯 줄이지만, 이 파일은
 * `ml/plan.ts`가 부르고 그 파일은 화면도 임포트한다 — 그러면 행렬 라이브러리가 **첫
 * 화면 번들**에 통째로 들어간다. 여기 있는 것은 전부 반복문이고, 그래서 이 모듈이
 * 딸려 보내는 것은 자기 자신뿐이다.
 */

import { uniformInt } from 'pure-rand/distribution/uniformInt'
import { xoroshiro128plus } from 'pure-rand/generator/xoroshiro128plus'

import { PCA_OVERSAMPLES, PCA_POWER_ITERATIONS } from '../limits'

/**
 * 배운 투영. **이것이 전처리기에 담겨 파일로 간다** (`ml/preprocess.ts`).
 *
 * 예측할 때도 같은 투영을 써야 하므로 저장한다 — 다시 맞추면 그 사진들의 평균으로
 * 다른 축을 얻고, 모델은 자기가 배운 적 없는 좌표계를 받는다.
 */
export interface Projection {
  /** 열마다의 평균. **투영 전에 뺀다** — 중심을 안 옮기면 첫 성분이 평균 방향이 된다. */
  readonly mean: readonly number[]
  /** `components[i]`가 i번째 주성분이고 길이는 원래 열 수다. 분산이 큰 순서다. */
  readonly components: readonly (readonly number[])[]
}

/** 0과 1 사이의 난수. 씨앗이 같으면 언제나 같다 (`ml/shuffle.ts`와 같은 난수원). */
function uniforms(seed: number): () => number {
  const rng = xoroshiro128plus(seed)
  // **`uniformInt`를 쓴다** — `ml/shuffle.ts`와 같은 문이다. 생성기의 날 정수를 직접
  // 꺼내면 구현마다 범위가 달라지고, 그때 같은 씨앗이 같은 그림을 못 준다.
  const WIDTH = 1 << 30
  return () => (uniformInt(rng, 0, WIDTH - 1) + 0.5) / WIDTH
}

/** 표준정규 난수 (Box-Muller). 무작위 SVD의 시험 행렬이 이것으로 찬다. */
function normals(seed: number): () => number {
  const next = uniforms(seed)
  return () => Math.sqrt(-2 * Math.log(next())) * Math.cos(2 * Math.PI * next())
}

/**
 * 열을 정규직교로 만든다 (수정 그람-슈미트).
 *
 * **고전 그람-슈미트를 쓰지 않는다.** 거듭제곱 반복을 지난 열들은 서로 거의 평행해서
 * 고전 식은 그 자리에서 직교성을 잃는다 — 그러면 아래 성분들이 같은 방향을 여러 번
 * 가리킨다.
 *
 * 길이가 0에 가까운 열은 버린다(돌려주는 열 수가 줄 수 있다). 그 열은 이미 앞의
 * 열들이 설명한 방향이다.
 */
function orthonormalize(matrix: Float64Array, rows: number, columns: number): number {
  let kept = 0
  for (let column = 0; column < columns; column += 1) {
    const target = kept
    if (target !== column) {
      for (let row = 0; row < rows; row += 1) {
        matrix[row * columns + target] = matrix[row * columns + column] ?? 0
      }
    }

    for (let done = 0; done < target; done += 1) {
      let dot = 0
      for (let row = 0; row < rows; row += 1) {
        dot += (matrix[row * columns + done] ?? 0) * (matrix[row * columns + target] ?? 0)
      }
      for (let row = 0; row < rows; row += 1) {
        matrix[row * columns + target] =
          (matrix[row * columns + target] ?? 0) - dot * (matrix[row * columns + done] ?? 0)
      }
    }

    let norm = 0
    for (let row = 0; row < rows; row += 1) {
      const value = matrix[row * columns + target] ?? 0
      norm += value * value
    }
    norm = Math.sqrt(norm)
    if (norm < 1e-10) continue

    for (let row = 0; row < rows; row += 1) {
      matrix[row * columns + target] = (matrix[row * columns + target] ?? 0) / norm
    }
    kept += 1
  }
  return kept
}

/**
 * 작은 대칭 행렬의 고윳값 분해 (야코비 회전).
 *
 * **작아서 이걸로 충분하다** — 여기 들어오는 것은 `(성분 수 + 여유)`의 제곱이라
 * 보통 60×60 안쪽이다. 큰 행렬에 쓸 방법은 아니지만 이 자리에는 과하지 않다.
 *
 * 돌려주는 것은 **고윳값이 큰 순서**로 정렬한 `{ values, vectors }`이고,
 * `vectors[i]`가 `values[i]`의 고유벡터다.
 */
function symmetricEigen(
  input: Float64Array,
  size: number,
): { values: number[]; vectors: number[][] } {
  const matrix = Float64Array.from(input)
  const basis = new Float64Array(size * size)
  for (let i = 0; i < size; i += 1) basis[i * size + i] = 1

  for (let sweep = 0; sweep < 60; sweep += 1) {
    let off = 0
    for (let p = 0; p < size; p += 1) {
      for (let q = p + 1; q < size; q += 1) {
        const value = matrix[p * size + q] ?? 0
        off += value * value
      }
    }
    if (off < 1e-20) break

    for (let p = 0; p < size; p += 1) {
      for (let q = p + 1; q < size; q += 1) {
        const apq = matrix[p * size + q] ?? 0
        if (Math.abs(apq) < 1e-15) continue
        const app = matrix[p * size + p] ?? 0
        const aqq = matrix[q * size + q] ?? 0
        const theta = (aqq - app) / (2 * apq)
        const sign = theta >= 0 ? 1 : -1
        const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const cos = 1 / Math.sqrt(t * t + 1)
        const sin = t * cos

        for (let i = 0; i < size; i += 1) {
          const ip = matrix[i * size + p] ?? 0
          const iq = matrix[i * size + q] ?? 0
          matrix[i * size + p] = cos * ip - sin * iq
          matrix[i * size + q] = sin * ip + cos * iq
        }
        for (let i = 0; i < size; i += 1) {
          const pi = matrix[p * size + i] ?? 0
          const qi = matrix[q * size + i] ?? 0
          matrix[p * size + i] = cos * pi - sin * qi
          matrix[q * size + i] = sin * pi + cos * qi
        }
        for (let i = 0; i < size; i += 1) {
          const ip = basis[i * size + p] ?? 0
          const iq = basis[i * size + q] ?? 0
          basis[i * size + p] = cos * ip - sin * iq
          basis[i * size + q] = sin * ip + cos * iq
        }
      }
    }
  }

  const order = Array.from({ length: size }, (_, index) => index).sort(
    (left, right) => (matrix[right * size + right] ?? 0) - (matrix[left * size + left] ?? 0),
  )
  return {
    values: order.map((index) => matrix[index * size + index] ?? 0),
    vectors: order.map((index) =>
      Array.from({ length: size }, (_, row) => basis[row * size + index] ?? 0),
    ),
  }
}

/**
 * 성분의 부호를 정한다 (sklearn의 `svd_flip`과 같은 규칙).
 *
 * **부호는 수학적으로 아무 뜻이 없다** — 성분에 -1을 곱해도 같은 축이다. 그래서
 * 규칙을 하나 정해 두지 않으면 같은 데이터가 실행마다 부호가 뒤집힌 좌표를 주고,
 * 학생은 그림이 뒤집힌 이유를 못 찾는다. **절댓값이 가장 큰 칸을 양수로 만든다.**
 */
function flipSigns(component: Float64Array): void {
  let at = 0
  let best = 0
  for (let index = 0; index < component.length; index += 1) {
    const size = Math.abs(component[index] ?? 0)
    if (size > best) {
      best = size
      at = index
    }
  }
  if ((component[at] ?? 0) < 0) {
    for (let index = 0; index < component.length; index += 1) {
      component[index] = -(component[index] ?? 0)
    }
  }
}

/**
 * 주성분 `count`개를 배운다. **못 배우면 `null`이고, 그때 투영은 일어나지 않는다.**
 *
 * `null`이 되는 자리는 셋이다 — 행이 없거나, 열이 없거나, **줄일 것이 없을 때**
 * (`count`가 이미 가진 열 수나 행 수 이상). 셋 다 "줄이지 않는다"가 정답이라 가르지
 * 않는다.
 *
 * **훈련 행만 들어온다.** 부르는 쪽(`ml/plan.ts`)이 분할 뒤의 훈련 행렬을 넘긴다 —
 * 테스트 행까지 보고 축을 정하면 스케일링 기준을 테스트에서 구하는 것과 같은 누수다
 * (`preprocess.previewBasisNote`가 학생에게 약속한 것).
 */
export function fitProjection(
  rows: readonly (readonly number[])[],
  count: number,
  seed: number,
): Projection | null {
  const n = rows.length
  const d = rows[0]?.length ?? 0
  if (n === 0 || d === 0) return null
  const k = Math.min(count, n, d)
  if (k < 1 || k >= d) return null

  // **중심을 옮긴 사본을 한 벌 만든다.** 아래에서 여러 번 훑으므로 그때마다 평균을
  // 빼면 같은 뺄셈을 수십 번 한다. `Float64Array` 한 벌이 n×d다.
  const mean = new Float64Array(d)
  for (const row of rows) {
    for (let j = 0; j < d; j += 1) mean[j] = (mean[j] ?? 0) + (row[j] ?? 0)
  }
  for (let j = 0; j < d; j += 1) mean[j] = (mean[j] ?? 0) / n

  const centered = new Float64Array(n * d)
  for (let i = 0; i < n; i += 1) {
    const row = rows[i] ?? []
    for (let j = 0; j < d; j += 1) centered[i * d + j] = (row[j] ?? 0) - (mean[j] ?? 0)
  }

  /** 시험 행렬의 폭. 여유분을 더해 뽑아야 끝쪽 성분이 안 흔들린다 (Halko §4.2). */
  const width = Math.min(k + PCA_OVERSAMPLES, n, d)
  const gauss = normals(seed)
  const sketch = new Float64Array(n * width)
  {
    const omega = new Float64Array(d * width)
    for (let index = 0; index < omega.length; index += 1) omega[index] = gauss()
    multiply(centered, omega, sketch, n, d, width)
  }

  /**
   * 거듭제곱 반복. **매번 직교화한다** — 안 하면 모든 열이 가장 큰 고유방향으로
   * 쏠려 수치적으로 같은 열이 된다.
   */
  const projected = new Float64Array(d * width)
  for (let round = 0; round < PCA_POWER_ITERATIONS; round += 1) {
    orthonormalize(sketch, n, width)
    multiplyTransposed(centered, sketch, projected, n, d, width)
    multiply(centered, projected, sketch, n, d, width)
  }
  const kept = orthonormalize(sketch, n, width)
  if (kept < 1) return null

  // B = Qᵀ·X — 작은 행렬(폭 × 열)이고, 여기부터는 원래 차원을 안 훑는다.
  const b = new Float64Array(kept * d)
  for (let column = 0; column < kept; column += 1) {
    for (let i = 0; i < n; i += 1) {
      const weight = sketch[i * width + column] ?? 0
      if (weight === 0) continue
      for (let j = 0; j < d; j += 1) {
        b[column * d + j] = (b[column * d + j] ?? 0) + weight * (centered[i * d + j] ?? 0)
      }
    }
  }

  // B·Bᵀ의 고유벡터가 곧 B의 왼쪽 특이벡터다. 작아서 야코비로 충분하다.
  const gram = new Float64Array(kept * kept)
  for (let p = 0; p < kept; p += 1) {
    for (let q = p; q < kept; q += 1) {
      let sum = 0
      for (let j = 0; j < d; j += 1) sum += (b[p * d + j] ?? 0) * (b[q * d + j] ?? 0)
      gram[p * kept + q] = sum
      gram[q * kept + p] = sum
    }
  }

  const { values, vectors } = symmetricEigen(gram, kept)
  const components: number[][] = []
  for (let index = 0; index < Math.min(k, kept); index += 1) {
    const scale = Math.sqrt(Math.max(values[index] ?? 0, 0))
    // 분산이 0인 방향은 성분이 될 수 없다. 거기서 멈춘다 — 뒤는 전부 0이다.
    if (scale < 1e-9) break
    const vector = vectors[index] ?? []
    const component = new Float64Array(d)
    for (let p = 0; p < kept; p += 1) {
      const weight = (vector[p] ?? 0) / scale
      if (weight === 0) continue
      for (let j = 0; j < d; j += 1) {
        component[j] = (component[j] ?? 0) + weight * (b[p * d + j] ?? 0)
      }
    }
    flipSigns(component)
    components.push(Array.from(component, rounded))
  }

  return components.length === 0 ? null : { mean: Array.from(mean, rounded), components }
}

/**
 * 파일에 적힐 자릿수로 자른다. **배우는 자리에서 자른다**(쓸 때가 아니라).
 *
 * **쓸 때 자르면 학습과 예측이 다른 수를 쓴다** — 학습은 안 자른 투영으로 돌고 예측은
 * 파일에서 읽은 자른 투영으로 돌아서, 같은 사진이 두 좌표를 받는다. 여기서 자르면
 * 두 자리가 **언제나 같은 수**를 본다.
 *
 * **왜 자르나.** 성분 하나가 1,280개의 수이고 배정도로 적으면 한 성분이 50KB다 —
 * 50개면 전처리기 하나가 2.5MB이고, 실험마다 한 벌씩 쌓인다. 여섯 자리면 네댓
 * 유효숫자가 남고(성분의 칸은 보통 0.0x다) 크기는 4분의 1이 된다.
 */
function rounded(value: number): number {
  return Math.round(value * 1e6) / 1e6
}

/** `out = left · right` — (n×d)·(d×w). */
function multiply(
  left: Float64Array,
  right: Float64Array,
  out: Float64Array,
  n: number,
  d: number,
  w: number,
): void {
  out.fill(0)
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < d; j += 1) {
      const value = left[i * d + j] ?? 0
      if (value === 0) continue
      for (let c = 0; c < w; c += 1) {
        out[i * w + c] = (out[i * w + c] ?? 0) + value * (right[j * w + c] ?? 0)
      }
    }
  }
}

/** `out = leftᵀ · right` — (d×n)·(n×w). */
function multiplyTransposed(
  left: Float64Array,
  right: Float64Array,
  out: Float64Array,
  n: number,
  d: number,
  w: number,
): void {
  out.fill(0)
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < d; j += 1) {
      const value = left[i * d + j] ?? 0
      if (value === 0) continue
      for (let c = 0; c < w; c += 1) {
        out[j * w + c] = (out[j * w + c] ?? 0) + value * (right[i * w + c] ?? 0)
      }
    }
  }
}

/**
 * 한 행을 주성분 좌표로 옮긴다. **학습도 예측도 이 함수를 지난다** —
 * 두 벌이 되면 예측만 다른 좌표계에 놓이고, 그림은 멀쩡해 보인다.
 */
export function applyProjection(projection: Projection, row: readonly number[]): number[] {
  const { mean, components } = projection
  return components.map((component) => {
    let sum = 0
    for (let j = 0; j < component.length; j += 1) {
      sum += ((row[j] ?? 0) - (mean[j] ?? 0)) * (component[j] ?? 0)
    }
    return sum
  })
}
