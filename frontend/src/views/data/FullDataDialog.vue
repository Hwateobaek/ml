<script setup lang="ts">
/**
 * 전체 데이터를 쪽으로 나눠 보는 팝업.
 *
 * **미리보기가 앞 스무 줄만 보여주기 때문에 있다** (`TABLE_PREVIEW_ROW_COUNT`). 학생이
 * 자기 데이터의 뒷부분을 아예 못 보면, 뒤에 섞여 있는 이상한 값도 못 본다.
 *
 * **그래도 통째로는 안 그린다** (`FULL_DATA_PAGE_SIZE`). 받는 표가 `MAX_DATASET_ROWS`
 * (십만)까지 오고, 열은 `MAX_DATASET_COLUMNS`(천)까지 온다 — 곱하면 칸이 억 단위이고,
 * 교실 PC는 거기서 멈춘다. `TabularPanel.vue`가 미리보기를 자르는 것과 같은 이유다.
 *
 * 자르는 계산은 `data/paging.ts`에 있다 (`tests/paging.spec.ts`가 지킨다).
 */

import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import AppButton from '@/components/AppButton.vue'
import AppDialog from '@/components/AppDialog.vue'
import AppTable from '@/components/AppTable.vue'
import { columnLabelsOf, type ColumnLabels } from '@/data/column-labels'
import { clampPage, pageCount, pageRange } from '@/data/paging'
import { FULL_DATA_PAGE_SIZE } from '@/limits'
import type { Dataset } from '@/ml/preprocess'

const props = defineProps<{
  open: boolean
  dataset: Dataset
  labels?: ColumnLabels | undefined
}>()

const emit = defineEmits<{ close: [] }>()

const { t } = useI18n()

const page = ref(1)

/** 다시 열면 처음부터 본다. 지난번에 보던 쪽에서 시작하면 어디인지 알 수 없다. */
watch(
  () => props.open,
  (open) => {
    if (open) page.value = 1
  },
)

const total = computed(() => props.dataset.rows.length)
const last = computed(() => pageCount(total.value, FULL_DATA_PAGE_SIZE))
const range = computed(() => pageRange(page.value, total.value, FULL_DATA_PAGE_SIZE))

const headers = computed(() => columnLabelsOf(props.dataset.columns, props.labels))

/** 이 쪽에 그릴 행들. **행 번호를 함께 준다** — 몇 번째 행을 보고 있는지가 곧 위치다. */
const rows = computed(() =>
  props.dataset.rows
    .slice(range.value.start, range.value.end)
    .map((cells, offset) => ({ number: range.value.start + offset + 1, cells })),
)

function go(delta: number): void {
  page.value = clampPage(page.value + delta, total.value, FULL_DATA_PAGE_SIZE)
}
</script>

<template>
  <AppDialog
    :open="props.open"
    size="wide"
    :title="t('data.tabular.fullDataTitle')"
    :description="
      t('data.tabular.fullDataDescription', {
        rows: total,
        columnCount: props.dataset.columns.length,
      })
    "
    @close="emit('close')"
  >
    <div class="flex min-h-0 flex-col gap-3">
      <!--
        **표만 굴린다** (`dialog-table-box`). 대화상자째 굴리면 아래로 내려가는 동안
        쪽 단추가 화면 밖으로 나가, 다음 쪽을 누르려고 도로 올라와야 한다.
      -->
      <AppTable class="dialog-table-box">
        <thead class="sticky top-0 z-10">
          <tr>
            <th>{{ t('data.tabular.rowNumber') }}</th>
            <th v-for="(name, index) in headers" :key="props.dataset.columns[index] ?? name">
              {{ name }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.number">
            <td class="tabular-nums text-ink-faint">{{ row.number }}</td>
            <!--
              **한 줄로 둔다.** 접히게 두면 `Chinstrap penguin (Pygoscelis antarctica)`
              같은 값 하나가 줄을 세 줄로 만들어, 한 쪽에 보이는 행이 셋으로 준다
              (2026-09-10, 사용자 화면). 넘치는 폭은 이 상자가 옆으로 굴려 보여준다.
            -->
            <td v-for="(cell, index) in row.cells" :key="index" class="whitespace-nowrap">
              {{ cell }}
            </td>
          </tr>
        </tbody>
      </AppTable>

      <!--
        **지금 어디를 보고 있는지를 행 번호로 말한다.** "3쪽"만으로는 그 쪽이 몇 번째
        행인지 알 수 없고, 학생이 찾는 것은 쪽이 아니라 행이다.
      -->
      <div class="flex flex-wrap items-center justify-between gap-3">
        <p class="text-ink-soft">
          {{
            t('data.tabular.fullDataRange', {
              first: range.start + 1,
              last: range.end,
              total,
            })
          }}
        </p>

        <div class="flex items-center gap-2">
          <AppButton variant="secondary" :disabled="page <= 1" @click="go(-1)">
            {{ t('common.prevPage') }}
          </AppButton>
          <span class="tabular-nums text-ink-soft">
            {{ t('data.tabular.pageOf', { page, last }) }}
          </span>
          <AppButton variant="secondary" :disabled="page >= last" @click="go(1)">
            {{ t('common.nextPage') }}
          </AppButton>
        </div>
      </div>
    </div>

    <template #actions>
      <AppButton variant="secondary" @click="emit('close')">{{ t('common.dismiss') }}</AppButton>
    </template>
  </AppDialog>
</template>
