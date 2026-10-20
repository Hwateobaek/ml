/**
 * TensorFlow.js 엔진이 받는 손잡이들. **이 엔진에서 이 값들의 유일한 출처다.**
 *
 * **엔진 본체와 파일이 갈라진 이유는 번들이다** (`mljs-params.ts`와 같은 사정). 학습
 * 화면이 손잡이를 그리려면 이 표를 읽어야 하는데, 엔진 본체에서 읽으면 `tfjs-layers`가
 * 통째로 첫 화면 번들에 딸려 온다 — 합성곱을 안 고른 학생이 그 값을 치르게 된다.
 * 여기에는 **의존성이 없다.**
 *
 * **이름은 케라스의 것이다** (CLAUDE.md §2 "파이썬 생태계의 관행을 따른다"). 교과서가
 * `filters`·`kernel_size`·`epochs`·`batch_size`를 쓰므로 화면에서 익힌 낱말이 교과서
 * 코드에서 그대로 보인다.
 *
 * **구조는 여기 없다.** 층을 쌓는 순서는 고정이고(open-decisions.md "합성곱을 직접
 * 학습하는 길을 연다") 학생이 고르는 것은 숫자뿐이다 — 그래서 이 표의 모양으로
 * 전부 담긴다.
 */

import type { HyperparameterSpec } from '../hyperparams'

export const TFJS_PARAMETERS: Readonly<Record<string, readonly HyperparameterSpec[]>> = {
  cnn: [
    /**
     * `[합성곱 → 풀링]`을 몇 번 반복하는가.
     *
     * **3이 기본이다.** 64×64로 들어와 풀링 세 번이면 8×8이 되고, 거기서 `Flatten`이
     * 받는 칸이 `8 × 8 × 필터 수`다.
     *
     * **적을수록 모델이 커진다.** 풀링을 덜 하면 `Flatten`에 들어오는 칸이 그만큼 많고,
     * 그 칸이 전부 완전연결층으로 간다 — 한 번만 접으면 가중치가 백만 개를 넘어
     * **파일에 담을 수 있는 크기(`MAX_MODEL_BYTES`)를 넘길 수 있다.** 그때도 학습은
     * 돌고 지표도 나오며, 모델만 안 담기고 그 사유가 파일에 남는다.
     *
     * **위쪽 끝이 4인 것은 눈금이 아니라 산수다** — 64를 반으로 네 번 접으면 4×4이고,
     * 그 아래로는 합성곱이 볼 것이 거의 없다.
     */
    { name: 'convBlocks', integer: true, min: 1, max: 4, step: 1, default: 3 },
    /**
     * 첫 합성곱층의 필터 수. **블록마다 두 배가 된다** — 16이면 16 → 32 → …이고,
     * 이것은 교과서가 보여 주는 VGG16의 모양이기도 하다(64 → 128 → 256).
     *
     * **깊어질수록 필터가 느는 이유**는 화면이 작아지면서 한 칸이 뜻하는 것이 복잡해지기
     * 때문이다. 학생에게는 손잡이 하나이고, 늘어나는 규칙은 화면이 설명한다.
     */
    { name: 'filters', integer: true, min: 1, max: 128, step: 1, default: 16 },
    /**
     * 커널 한 변의 길이. **3이 기본이다** — 교과서의 VGG16이 전부 3×3이고, 작은 커널을
     * 여러 겹 쌓는 것이 큰 커널 한 겹보다 낫다는 것이 그 구조의 요지다.
     *
     * **짝수도 받는다.** 2×2 커널은 흔치 않지만 고장이 아니다 — 막는 대신 눈금이
     * 홀수 쪽에 있게 둔다.
     */
    { name: 'kernelSize', integer: true, min: 2, max: 7, step: 1, default: 3 },
    /**
     * 합성곱 뒤에 오는 완전연결층의 뉴런 수. **교과서의 `Dense(64, activation='relu')`가
     * 이 칸이다** (151쪽).
     *
     * **인공신경망의 `neuronsPerLayer`와 같은 뜻이고 다른 자리다** — 저쪽은 백본이 만든
     * 1,280개 위에 서고, 여기는 **학생이 쌓은 합성곱층이 만든 것** 위에 선다.
     */
    { name: 'denseUnits', integer: true, min: 1, max: 512, step: 1, default: 64 },
    /**
     * 데이터를 몇 바퀴 도는가. **교과서의 `epochs=5`가 기본이다** (151쪽의 `model.fit`).
     *
     * **인공신경망의 200과 다르다.** 저쪽은 sklearn `MLPClassifier`의 기본값이고 수렴하면
     * 스스로 멈추는데, 여기는 **설정한 바퀴를 다 돈다** — 케라스가 그렇고 교과서가
     * 보여 주는 것도 그것이다. 그래서 숫자가 곧 시간이다.
     */
    { name: 'epochs', integer: true, min: 1, max: 100, step: 1, default: 5 },
    /** 한 번에 몇 장씩 보는가. **교과서의 `batch_size=32` 그대로다.** */
    { name: 'batchSize', integer: true, min: 1, max: 256, step: 1, default: 32 },
    /**
     * 한 걸음의 크기. **케라스 `Adam`의 기본값 0.001이다** — 교과서가
     * `optimizer='adam'`이라고만 적고 넘어가는 그 값이다.
     */
    { name: 'learningRate', integer: false, min: 0.0001, max: 1, step: 0.0001, default: 0.001 },
  ],
}
