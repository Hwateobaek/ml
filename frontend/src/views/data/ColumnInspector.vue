<script setup lang="ts">
/**
 * 열 검사기. 열마다 자료형·결측 수·값 종류·예시를 늘어놓는다.
 *
 * **넓은 화면에서는 표 옆에 서고 좁은 화면에서는 표 아래에 접힌다** (architecture.md §8.9).
 * 그래서 자리를 스스로 정하지 않는다 — 바깥이 준 만큼 채우고, 넘치면 **자기 안에서**
 * 스크롤한다. 여기서 높이를 박으면 두 자리 중 한쪽에서 반드시 틀린다.
 *
 * **이름을 고치는 자리이기도 하다.** 고치는 것은 **부르는 이름뿐이고 정본은 안 바뀐다**
 * (`data/column-labels.ts`). 그래서 고친 뒤에도 타깃·특성은 그대로 붙어 있다.
 *
 * **확정한 데이터에서만 고친다** (`editable`). 아직 확정하지 않은 미리보기에서 고치면
 * 적어 둘 곳이 없다 — 대응표는 프로젝트 설정에 사는데 그 표는 아직 프로젝트가 아니다.
 *
 * 판단은 하나도 안 한다. `data/columns.ts`가 요약한 것을 그리고, 겹치는 이름인지는
 * `data/column-labels.ts`가 답한다.
 */

import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import AppTable from '@/components/AppTable.vue'
import { columnLabel, columnLabelProblem, type ColumnLabels } from '@/data/column-labels'
import type { ColumnSummary } from '@/data/columns'

const props = defineProps<{
  columns: readonly ColumnSummary[]
  /** 없으면 아무것도 안 고친 것과 같다 — 이미지·미리보기가 그 자리다. */
  labels?: ColumnLabels | undefined
  /** 참이면 이름 칸이 입력칸이 된다. */
  editable?: boolean
  /**
   * 열마다 이상치 개수 (`data/visualize.ts`의 `outlierCounts`). **확정한 표로만 센다** —
   * 미리보기는 앞 스무 줄뿐이라 사분위수가 표 전체의 것이 아니다. 없으면 이 칸이 통째로
   * 빈 표시다.
   */
  outliers?: ReadonlyMap<string, number> | undefined
}>()

const emit = defineEmits<{ rename: [original: string, next: string] }>()

const { t } = useI18n()

/**
 * 겹쳐서 못 쓴 이름. **키는 원본 이름이다** — 화면에 보이는 이름은 방금 학생이 친
 * 글자라 아직 아무것도 가리키지 않는다.
 */
const rejected = ref<Record<string, true>>({})

const shownName = (column: ColumnSummary): string => columnLabel(column.name, props.labels)

/** 숫자가 없는 열과 아직 안 센 표는 0이 아니라 빈 표시다 (`ColumnPicker.vue`와 같다). */
function outlierText(column: ColumnSummary): string {
  const count = props.outliers?.get(column.name)
  return count === undefined ? '–' : String(count)
}

/**
 * 입력칸을 떠날 때 확정한다. **글자마다 확정하지 않는다** — 치는 도중의 `부`·`부리`가
 * 전부 저장이 되고, 그 사이 값들이 자동 저장을 깨운다.
 */
function commit(column: ColumnSummary, event: Event): void {
  const input = event.target
  if (!(input instanceof HTMLInputElement)) return
  const next = input.value
  const names = props.columns.map((each) => each.name)

  if (columnLabelProblem(names, props.labels, column.name, next) !== null) {
    rejected.value = { ...rejected.value, [column.name]: true }
    // 되돌려 놓는다. 안 그러면 못 쓴 이름이 칸에 남아 저장된 것처럼 보인다.
    input.value = shownName(column)
    return
  }

  const rest = { ...rejected.value }
  delete rest[column.name]
  rejected.value = rest
  emit('rename', column.name, next)
}
</script>

<template>
  <AppTable class="min-h-0 flex-1">
    <thead class="sticky top-0 z-10">
      <tr>
        <th>{{ t('data.tabular.columnName') }}</th>
        <th>{{ t('data.tabular.kind') }}</th>
        <th>{{ t('data.tabular.missing') }}</th>
        <th>{{ t('data.tabular.outliers') }}</th>
        <th>{{ t('data.tabular.unique') }}</th>
        <th>{{ t('data.tabular.samples') }}</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="column in props.columns" :key="column.name">
        <td class="font-bold">
          <template v-if="props.editable">
            <input
              :value="shownName(column)"
              type="text"
              :aria-label="t('data.tabular.renameLabel', { name: column.name })"
              class="w-full rounded-field border border-line-strong bg-surface px-2 py-1 font-bold"
              @change="commit(column, $event)"
              @keyup.enter="commit(column, $event)"
            />
            <!--
              **원본 이름은 고친 뒤에만 보인다.** 안 고쳤으면 입력칸에 그 이름이 이미
              있어서, 같은 글자를 두 번 찍는 것이 된다.
            -->
            <p v-if="shownName(column) !== column.name" class="mt-1 font-normal text-ink-faint">
              {{ column.name }}
            </p>
            <p v-if="rejected[column.name]" class="mt-1 font-medium text-danger">
              {{ t('data.tabular.renameDuplicate') }}
            </p>
          </template>
          <template v-else>{{ shownName(column) }}</template>
        </td>
        <td>{{ t(`columnKind.${column.kind}`) }}</td>
        <td>{{ column.missing }}</td>
        <td>{{ outlierText(column) }}</td>
        <td>{{ column.unique }}</td>
        <td class="text-ink-soft">{{ column.samples.join(', ') }}</td>
      </tr>
    </tbody>
  </AppTable>
</template>
