<script setup lang="ts">
/**
 * 이상치 처리의 `range` — 열마다 쓸 범위를 적는 칸들
 * (open-decisions.md "이상치는 훈련 데이터의 IQR로 클리핑한다"의 "사람이 정한 범위로는 행을 뺀다").
 *
 * **판단은 여기 없다.** 글자를 숫자로 읽는 것은 `parseBound`, 뒤집혔는지는 `rangeProblem`,
 * 옆의 참고값은 `rangeGuide`가 정한다 (`tests/ranges.spec.ts`가 지킨다). 이 부품은 그것을
 * 그리고, 쓸 수 있는 범위만 올려 보낸다.
 *
 * **스토어를 안 본다.** 판이 열 목록과 대응표를 내려준다 — `ColumnPicker`가 단독으로 마운트해
 * 검사되는 것과 같은 모양으로 둔다.
 */

import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import { useFormat } from '@/composables/useFormat'
import { columnLabel, type ColumnLabels } from '@/data/column-labels'
import type { RangeGuide } from '@/data/visualize'
import { parseBound, rangeProblem, type ColumnRange, type ColumnRanges } from '@/ml/ranges'

const props = defineProps<{
  /**
   * 범위를 정할 수 있는 열. **지금 쓰는 숫자 열뿐이다** — 특성에서 뺀 열의 범위는 행을 안
   * 빼므로(`appliedRanges`) 칸을 보이면 효과 없는 입력을 받는 셈이다.
   */
  columns: readonly { readonly name: string; readonly guide: RangeGuide | null }[]
  ranges: ColumnRanges | undefined
  labels?: ColumnLabels | undefined
}>()

const emit = defineEmits<{ set: [column: string, range: ColumnRange] }>()

const { t } = useI18n()
const format = useFormat()

/** 칸마다 못 쓴 이유. **키는 원본 열 이름이다.** 고치면 사라진다. */
const problems = ref<Record<string, 'reversed' | 'notNumber'>>({})

function shown(value: number | undefined): string {
  return value === undefined ? '' : String(value)
}

/**
 * 칸을 떠날 때 확정한다. **글자마다 확정하지 않는다** — `1`을 치는 순간 행이 빠지고 `15`에서
 * 돌아오면 그 사이에 자동 저장과 전처리 계획이 두 번 돈다.
 *
 * **못 쓰는 입력은 파일에 안 간다.** 숫자가 아니거나 뒤집혔으면 이유만 보이고, 칸은 학생이
 * 친 글자를 그대로 둔다 — 지우면 무엇을 잘못 쳤는지 볼 수 없다.
 */
function commit(column: string, side: 'min' | 'max', event: Event): void {
  const input = event.target
  if (!(input instanceof HTMLInputElement)) return
  const parsed = parseBound(input.value)
  const rest = { ...problems.value }
  delete rest[column]

  if (parsed === null) {
    problems.value = { ...rest, [column]: 'notNumber' }
    return
  }
  const next: ColumnRange = { ...(props.ranges?.[column] ?? {}), [side]: parsed }
  if (rangeProblem(next) !== null) {
    problems.value = { ...rest, [column]: 'reversed' }
    return
  }
  problems.value = rest
  emit('set', column, next)
}
</script>

<template>
  <div class="flex flex-col gap-3">
    <p class="text-ink-soft">{{ t('preprocess.tabular.rangesLead') }}</p>

    <p v-if="props.columns.length === 0" class="text-ink-faint">
      {{ t('preprocess.tabular.rangesEmpty') }}
    </p>

    <!--
      **열마다 한 덩어리다.** 이름 → 두 칸 → 참고값 → 이유. 참고값이 칸 바로 아래 있어야
      학생이 시각화 화면에서 본 경계를 옮겨 적을 수 있다.
    -->
    <div
      v-for="column in props.columns"
      :key="column.name"
      class="flex flex-col gap-1.5 border-t border-line pt-3"
    >
      <span class="font-bold">{{ columnLabel(column.name, props.labels) }}</span>
      <div class="flex flex-wrap items-center gap-2">
        <label class="flex items-center gap-2">
          <span class="text-ink-soft">{{ t('preprocess.tabular.rangeMin') }}</span>
          <input
            type="text"
            inputmode="decimal"
            class="w-28 rounded-field border border-line-strong bg-surface px-2 py-1 tabular-nums"
            :value="shown(props.ranges?.[column.name]?.min)"
            :aria-label="
              t('preprocess.tabular.rangeMinLabel', {
                name: columnLabel(column.name, props.labels),
              })
            "
            @change="commit(column.name, 'min', $event)"
            @keyup.enter="commit(column.name, 'min', $event)"
          />
        </label>
        <label class="flex items-center gap-2">
          <span class="text-ink-soft">{{ t('preprocess.tabular.rangeMax') }}</span>
          <input
            type="text"
            inputmode="decimal"
            class="w-28 rounded-field border border-line-strong bg-surface px-2 py-1 tabular-nums"
            :value="shown(props.ranges?.[column.name]?.max)"
            :aria-label="
              t('preprocess.tabular.rangeMaxLabel', {
                name: columnLabel(column.name, props.labels),
              })
            "
            @change="commit(column.name, 'max', $event)"
            @keyup.enter="commit(column.name, 'max', $event)"
          />
        </label>
      </div>
      <p v-if="column.guide" class="text-ink-faint tabular-nums">
        {{
          column.guide.bounds
            ? t('preprocess.tabular.rangeGuide', {
                min: format.stat(column.guide.min),
                max: format.stat(column.guide.max),
                low: format.stat(column.guide.bounds.low),
                high: format.stat(column.guide.bounds.high),
              })
            : t('preprocess.tabular.rangeGuideFlat', {
                min: format.stat(column.guide.min),
                max: format.stat(column.guide.max),
              })
        }}
      </p>
      <p v-if="problems[column.name] === 'reversed'" class="font-medium text-danger">
        {{ t('preprocess.tabular.rangeReversed') }}
      </p>
      <p v-else-if="problems[column.name] === 'notNumber'" class="font-medium text-danger">
        {{ t('preprocess.tabular.rangeNotNumber') }}
      </p>
    </div>
  </div>
</template>
