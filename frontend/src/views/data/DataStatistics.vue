<script setup lang="ts">
/**
 * 데이터 통계. **수치형 열마다** 개수·평균·표준편차·최솟값·사분위수·최댓값을 한 표에 둔다.
 *
 * **행이 통계이고 열이 특성이다** — 교과서의 통계표(pandas `describe()`)와 같은 방향이라
 * 학생이 둘을 나란히 놓고 맞춰 볼 수 있다. 열이 많으면 `AppTable`이 옆으로 스크롤한다.
 *
 * **확정한 표 전체로 센다.** 미리보기의 앞 스무 줄로 구하면 평균이 표 전체의 것이 아니다.
 * 계산은 `data/visualize.ts`의 `descriptiveStats`가 한다.
 */

import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import AppTable from '@/components/AppTable.vue'
import { useFormat } from '@/composables/useFormat'
import { columnLabel, type ColumnLabels } from '@/data/column-labels'
import type { ColumnSummary } from '@/data/columns'
import { descriptiveStats, numericValues, type DescriptiveStats } from '@/data/visualize'
import type { Dataset } from '@/ml/preprocess'

const props = defineProps<{
  dataset: Dataset
  columns: readonly ColumnSummary[]
  labels?: ColumnLabels | undefined
}>()

const { t } = useI18n()
const format = useFormat()

/** 표의 행. **키를 조립하지 않고 적어 둔다** — 조립하면 로케일 검사가 짝을 못 본다. */
const STAT_ROWS = [
  { stat: 'count', label: 'data.tabular.statCount' },
  { stat: 'mean', label: 'data.tabular.statMean' },
  { stat: 'std', label: 'data.tabular.statStd' },
  { stat: 'min', label: 'data.tabular.statMin' },
  { stat: 'q1', label: 'data.tabular.statQ1' },
  { stat: 'median', label: 'data.tabular.statMedian' },
  { stat: 'q3', label: 'data.tabular.statQ3' },
  { stat: 'max', label: 'data.tabular.statMax' },
] as const

const stats = computed(() =>
  props.columns
    .filter((column) => column.kind === 'numeric')
    .map((column) => ({
      name: column.name,
      stats: descriptiveStats(numericValues(props.dataset, column.name)),
    })),
)

function cell(stats: DescriptiveStats | null, row: (typeof STAT_ROWS)[number]['stat']): string {
  if (stats === null) return '–'
  if (row === 'count') return String(stats.count)
  const value = stats[row]
  return value === null ? '–' : format.describe(value)
}
</script>

<template>
  <section class="flex min-w-0 flex-col gap-3 rounded-panel border border-line bg-surface p-4">
    <h3 class="leading-tight font-bold text-ink-soft">{{ t('data.tabular.statistics') }}</h3>
    <p v-if="stats.length === 0" class="text-base text-ink-faint">
      {{ t('data.tabular.statisticsNone') }}
    </p>
    <AppTable v-else>
      <thead>
        <tr>
          <th>{{ t('data.tabular.statisticsHead') }}</th>
          <th v-for="column in stats" :key="column.name" class="text-right whitespace-nowrap">
            {{ columnLabel(column.name, props.labels) }}
          </th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in STAT_ROWS" :key="row.stat">
          <th class="whitespace-nowrap">{{ t(row.label) }}</th>
          <td v-for="column in stats" :key="column.name" class="text-right whitespace-nowrap">
            {{ cell(column.stats, row.stat) }}
          </td>
        </tr>
      </tbody>
    </AppTable>
  </section>
</template>
