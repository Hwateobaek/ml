<script setup lang="ts">
/**
 * 입력 하나를 감싸는 껍데기 — 라벨, 도움말, 오류.
 *
 * **라벨과 입력을 잇는 id를 컴포넌트가 만든다.** 화면마다 손으로 붙이면 언젠가
 * 빠뜨리고, 빠진 라벨은 눈으로는 멀쩡해 보인다. 슬롯으로 내려주는 값을 그대로 쓴다.
 *
 * ```
 * <AppField :label="t('data.tabular.fileName')">
 *   <template #default="field"><input v-bind="field" /></template>
 * </AppField>
 * ```
 *
 * 오류가 있으면 도움말 자리를 오류가 차지한다. 둘을 같이 띄우면 어느 쪽을 읽어야
 * 하는지 알 수 없다.
 *
 * **순서는 이름 → 도움말 → 입력이다.** 도움말이 입력 아래에 있으면 **읽기 전에 이미
 * 답하고 있다** — 이름을 다 적고 나서야 "파일 이름이 됩니다"를 만난다. 그리고 축을
 * 고르는 칸(`AppChoices`)은 그 한 줄이 위에 있어서, 나란히 선 두 칸의 순서가 서로
 * 달랐다.
 */

import { computed, useId } from 'vue'

const props = defineProps<{
  label: string
  /**
   * 이 칸이 **무엇인지** 설명하는 한 문장. 값과 무관하게 늘 같다.
   *
   * **아래 `hint`와 자리가 다르다** — 저쪽은 지금 상태(쓸 수 있는 범위·오류)를 말하고
   * 오류가 나면 오류에 자리를 내준다. 설명은 그때도 남아야 한다: 범위를 벗어난 학생이
   * 읽을 것이 "이 값이 무엇인가"이기 때문이다 (`hyperparamHelp.*`).
   */
  help?: string | undefined
  hint?: string | undefined
  /** 있으면 도움말 자리를 차지한다. 조건부로 넘기는 자리라 undefined를 받는다. */
  error?: string | undefined
}>()

const inputId = useId()
const noteId = useId()
const helpId = useId()

const note = computed(() => props.error ?? props.hint)

/**
 * 입력이 가리키는 설명들. **둘 다 가리킨다** — 설명과 상태가 다른 문장이라
 * 스크린 리더가 하나만 읽으면 나머지 하나는 없는 것과 같다.
 */
const describedBy = computed(
  () =>
    [props.help === undefined ? null : helpId, note.value === undefined ? null : noteId]
      .filter((id) => id !== null)
      .join(' ') || undefined,
)
</script>

<template>
  <div class="flex flex-col gap-1.5">
    <label :for="inputId" class="text-base font-bold text-ink-soft">{{ label }}</label>

    <!--
      **설명이 먼저다.** 이름 → 설명 → 상태 → 입력 — 데이터 정제 카드가 선택지마다
      쓰는 차례와 같다 (`TabularPrepPanel`). 설명을 입력 아래에 두면 학생이 값을 고친
      뒤에야 그 값이 무엇인지 읽는다.
    -->
    <p v-if="help !== undefined" :id="helpId" class="text-base text-ink-soft">{{ help }}</p>

    <p
      v-if="note !== undefined"
      :id="noteId"
      class="text-base"
      :class="error === undefined ? 'text-ink-faint' : 'font-medium text-danger'"
    >
      {{ note }}
    </p>

    <slot :id="inputId" :aria-describedby="describedBy" :aria-invalid="error !== undefined" />
  </div>
</template>
