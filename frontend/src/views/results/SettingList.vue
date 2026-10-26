<script setup lang="ts">
/**
 * 이 실험이 **실제로 쓴 설정**. 평가하는 사람이 읽는 자리다.
 *
 * **옆의 `ChangeList`와 묻는 것이 다르다** — 저쪽은 *"무엇이 바뀌었나"*이고 여기는
 * *"무엇으로 했나"*다. 첫 실험이거나 설정을 안 바꾸고 다시 학습하면 저쪽은 비는데,
 * 그때도 과정은 평가되어야 한다 (CLAUDE.md §1.3 — 교사는 파일 하나로 다 본다).
 *
 * **값의 어휘는 `ml/changes.ts`가 낸다.** 여기서 문장을 짓지 않는다 — 두 화면이 같은
 * 등록부를 지나므로 *"바뀐 것"*과 *"설정"*이 같은 낱말로 말한다.
 */

import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import AppBadge from '@/components/AppBadge.vue'
import AppPopover from '@/components/AppPopover.vue'
import { useFormat } from '@/composables/useFormat'
import { ACTION_ICONS } from '@/icons'
import type { ChangeValue, Setting } from '@/ml/changes'

const props = defineProps<{ settings: readonly Setting[] }>()

const { t } = useI18n()
const format = useFormat()

/**
 * 값 하나를 문자열로. **`ChangeList`와 같은 규칙이다** — 어휘는 로케일에서, 개수는
 * 단위와 함께, 비율은 백분율로.
 */
function valueText(value: ChangeValue): string {
  if (value.kind === 'locale') return t(value.key)
  if (value.kind === 'count') return t('meta.countUnit', { count: value.count })
  if (value.kind === 'literal') return value.text
  if (value.kind === 'ratio') return format.percent(value.value)
  return t('meta.none')
}

/**
 * 목록 항목 하나를 사람이 읽는 이름으로. **`ChangeList`의 같은 함수와 규칙이 같다** —
 * 열 이름은 학생의 낱말이라 그대로, 모델 식별자는 로케일에서 찾는다.
 */
function itemText(kind: Setting['value'], item: string): string {
  const itemKind = kind.kind === 'count' ? (kind.itemKind ?? 'text') : 'text'
  if (itemKind !== 'model') return item
  const [algorithm, runtime] = item.split(':')
  if (!algorithm || !runtime) return item
  return t('predict.modelName', {
    algorithm: t(`algorithms.${algorithm}`),
    runtime: t(`runtimes.${runtime}`),
  })
}

/**
 * 줄마다 한 번만 센다. **템플릿에서 부르면 렌더마다 다시 돈다** (`ChangeList`가
 * V11 R5 C-3에서 같은 것을 고쳤다).
 */
const rows = computed(() =>
  props.settings.map((setting) => ({
    setting,
    /** 이름까지 아는 목록인가. **개수만으로는 어느 열을 썼는지 알 수 없다.** */
    items:
      setting.value.kind === 'count' && setting.value.items && setting.value.items.length > 0
        ? setting.value.items.map((name) => itemText(setting.value, name))
        : null,
  })),
)
</script>

<template>
  <!--
    **정의 목록이다.** 이름과 값의 짝이 줄마다 하나이고 그 사이에 순서나 우열이 없다 —
    `ChangeList`가 불릿을 쓰는 것과 갈리는 자리다(저쪽은 "일어난 일"의 목록이라 문장이고,
    여기는 "이름: 값"이라 표에 가깝다).
  -->
  <dl class="flex flex-col gap-1.5">
    <div
      v-for="{ setting, items } in rows"
      :key="setting.path"
      class="flex flex-wrap items-baseline gap-x-2 gap-y-1.5"
    >
      <dt>
        <AppBadge>{{ t(setting.labelKey) }}</AppBadge>
      </dt>
      <dd class="flex flex-wrap items-baseline gap-x-2 gap-y-1.5">
        <span>{{ valueText(setting.value) }}</span>

        <!--
          **개수만으로는 무엇을 썼는지 알 수 없다.** 평가에서 가장 알고 싶은 것이
          *"어느 열을 특성으로 골랐나"*인데, 이름을 줄에 늘어놓으면 특성 스무 개가
          화면을 덮는다 — 그래서 눌러야 나온다 (`ChangeList`와 같은 규칙).
        -->
        <AppPopover v-if="items" size="wide" side="top">
          <template #trigger="{ open }">
            <button
              type="button"
              :aria-expanded="open"
              class="flex items-center gap-1 rounded-control text-brand underline decoration-dotted underline-offset-4 transition-colors hover:text-brand-strong"
            >
              {{ t('results.settingItems') }}
              <component :is="ACTION_ICONS.explainTerm" :size="16" aria-hidden="true" />
            </button>
          </template>

          <h4 class="font-bold text-ink">{{ t(setting.labelKey) }}</h4>
          <ul class="mt-2 flex flex-wrap gap-1.5">
            <li v-for="name in items" :key="name">
              <AppBadge>{{ name }}</AppBadge>
            </li>
          </ul>
        </AppPopover>
      </dd>
    </div>
  </dl>
</template>
