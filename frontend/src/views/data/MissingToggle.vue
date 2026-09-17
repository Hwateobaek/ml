<script setup lang="ts">
/**
 * [결측치 표시] 단추와 그 범례. 데이터 미리보기와 전체 데이터 보기가 함께 쓴다.
 *
 * **켜고 끄는 상태는 바깥이 갖는다** (`TabularPanel.vue`). 미리보기에서 켜고 팝업을 열면
 * 거기서도 켜져 있어야 한다 — 같은 표를 보는 두 창이 다르게 칠해져 있으면 학생은 어느
 * 쪽이 맞는지 묻는다.
 *
 * **숫자는 전체 표의 것이다.** 미리보기에는 앞 스무 줄뿐이라, 거기서 센 수를 적으면 뒤에
 * 있는 결측 행이 없는 것처럼 읽힌다.
 */

import { useI18n } from 'vue-i18n'

import AppButton from '@/components/AppButton.vue'

const props = defineProps<{
  on: boolean
  /** 전체 표에서 결측치가 있는 행의 수. 확정하기 전의 표에는 없다. */
  count?: number | undefined
}>()

const emit = defineEmits<{ toggle: [] }>()

const { t } = useI18n()
</script>

<template>
  <div class="flex flex-wrap items-center gap-x-3 gap-y-1">
    <AppButton variant="secondary" :aria-pressed="props.on" @click="emit('toggle')">
      {{ props.on ? t('data.tabular.missingHide') : t('data.tabular.missingShow') }}
    </AppButton>
    <!-- 범례는 켰을 때만 선다. 꺼져 있을 때 색 설명은 가리킬 색이 없다. -->
    <p v-if="props.on" class="flex flex-wrap items-center gap-x-3 gap-y-1 text-ink-soft">
      <span class="flex items-center gap-1.5">
        <span class="inline-block size-3.5 rounded-sm border border-caution/40 bg-caution-soft" />
        {{
          props.count === undefined
            ? t('data.tabular.missingRowLegend')
            : t('data.tabular.missingRowCount', { count: props.count })
        }}
      </span>
      <span class="flex items-center gap-1.5">
        <span class="inline-block size-3.5 rounded-sm border border-danger/40 bg-danger-soft" />
        {{ t('data.tabular.missingCellLegend') }}
      </span>
    </p>
  </div>
</template>
