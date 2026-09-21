<script setup lang="ts">
/**
 * 모델이 배운 값 — 계수·절편, 그리고 범주별 평균·분산.
 *
 * **어느 모델이 무엇을 보여주는지는 여기가 아니라 `ml/parameters.ts`와 등록부에 있다**
 * (`ml/metric-panels.ts`, architecture.md §9.1). 이 파일은 "어떻게 그리는가"만 안다.
 *
 * **표는 특성이 세로다.** 원핫이면 열 하나가 여럿으로 늘어나 특성이 수십 줄이 되는데,
 * 가로로 두면 그만큼 옆으로 흐르고 좁은 화면에서 읽을 수 없다. 범주는 보통 둘셋이라
 * 가로가 맞다.
 */

import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import AppTable from '@/components/AppTable.vue'
import { useFormat } from '@/composables/useFormat'
import type { PanelInput } from '@/ml/metric-panels'
import { PARAMETER_TITLE_KEYS, parameterTableFor } from '@/ml/parameters'

const props = defineProps<{ input: PanelInput }>()

const { t } = useI18n()
const format = useFormat()

/**
 * 그릴 것 전부. **못 세우면 `null`이고 그때 이 패널은 아무것도 안 그린다**
 * (§9.2 "없는 것을 이름으로 말하지 않는다"). 전처리기가 안 담긴 파일이 그렇다 —
 * 이름 없이 숫자만 늘어놓으면 몇 번째 계수가 어느 열인지 알 수 없다.
 */
const table = computed(() => {
  const { run, modelBytes, preprocessor, experiment } = props.input
  return parameterTableFor(run.model?.format, modelBytes, preprocessor, experiment.settings)
})

/**
 * 배운 값으로 세운 식 한 줄 — `mass = 42.53 × neck + 67.76`.
 *
 * **부호와 자릿수를 정하는 자리가 여기다** (`ml/parameters.ts`는 계수를 그대로 준다).
 * 음수 계수는 `+ -3.2`가 아니라 `− 3.2`로 적고, **맨 앞 항만 `+`를 안 붙인다.**
 * 수는 다른 화면과 같은 유효숫자로 자른다(`format.stat`) — 식만 자릿수가 다르면 위 표의
 * 계수와 다른 값처럼 보인다.
 *
 * **특성이 하나도 없으면 상수만 남는다.** 그때 `= + 67.76`이라고 적으면 식이 아니다.
 */
const equationText = computed(() => {
  const equation = table.value?.equation
  if (!equation) return null

  const head = equation.target ?? t('results.equationTarget')
  const term = (value: number, first: boolean): string => {
    const size = format.stat(Math.abs(value))
    if (first) return value < 0 ? `−${size}` : size
    return value < 0 ? ` − ${size}` : ` + ${size}`
  }

  const terms = equation.terms
    .map((one, index) => `${term(one.coefficient, index === 0)} × ${one.name}`)
    .join('')
  return `${head} = ${terms}${term(equation.intercept, terms === '')}`
})
</script>

<template>
  <section v-if="table" class="flex min-w-0 flex-col gap-5">
    <!--
      **머리와 표는 다른 덩어리다.** 카드 본문이 덩어리를 `gap-5`로 떼고 덩어리 안에서
      제목과 내용을 `gap-1.5`로 붙인다(`RunDetail`·`ClusterResultPanel`). 처음에 전부
      `gap-1.5`로 두었더니 표 제목이 설명문에 붙어 어느 쪽에 속한 줄인지 흐려졌다.
    -->
    <div class="flex flex-col gap-1.5">
      <h4 class="font-bold">{{ t('results.parametersTitle') }}</h4>
      <p class="text-muted">{{ t('results.parametersLead') }}</p>
      <!--
      **스케일링을 켰으면 안 뜬다.** 계수의 크기를 견줄 수 있는지가 거기서 갈리고,
      켠 학생에게까지 띄우면 맞는 말을 못 믿게 만든다.
    -->
      <p v-if="!table.scaled" class="text-caution">{{ t('results.parametersScaleCaution') }}</p>
    </div>

    <div
      v-for="section in table.sections"
      :key="section.kind"
      class="flex min-w-0 flex-col gap-1.5"
    >
      <h5 class="font-bold">{{ t(PARAMETER_TITLE_KEYS[section.kind]) }}</h5>

      <AppTable>
        <thead>
          <tr>
            <th>{{ t('results.parametersFeature') }}</th>
            <!--
              **범주 이름이 곧 열 이름이다.** 회귀는 범주가 없어 `label`이 비고,
              그때는 열이 하나뿐이라 `값`이라고만 적는다.
            -->
            <th v-for="(row, index) in section.rows" :key="index">
              {{ row.label ?? t('results.parametersValue') }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(name, column) in table.featureNames" :key="name">
            <th class="text-left">{{ name }}</th>
            <td v-for="(row, index) in section.rows" :key="index">
              {{ format.stat(row.values[column] ?? 0) }}
            </td>
          </tr>
          <!--
            **절편은 특성이 아니라서 표 안에 섞지 않는다.** 평균·분산 표에는 없다.
          -->
          <tr v-if="section.rows.some((row) => row.intercept !== null)">
            <th class="text-left">{{ t('results.parametersIntercept') }}</th>
            <td v-for="(row, index) in section.rows" :key="index">
              {{ row.intercept === null ? '' : format.stat(row.intercept) }}
            </td>
          </tr>
        </tbody>
      </AppTable>
    </div>

    <!--
      **배운 값으로 세운 식.** 표가 계수와 절편을 따로 보여주는데, 그것을 하나의 식으로
      잇는 일을 화면이 대신한다 — 그 식이 곧 "예측을 어떻게 계산하는가"다.

      **표 뒤에 선다.** 문구가 *"위의 계수와 절편으로 세운 식"*이라고 말하므로, 앞에 두면
      화면이 자기 자리에 대해 거짓말을 한다.

      **부호와 자릿수는 화면이 정한다** (`ml/parameters.ts`는 계수를 그대로 준다).
    -->
    <div v-if="equationText" class="flex min-w-0 flex-col gap-1.5">
      <h5 class="font-bold">{{ t('results.equationTitle') }}</h5>
      <p class="text-muted">{{ t('results.equationLead') }}</p>
      <!--
        **가로로 넘치면 이 상자 안에서 굴린다.** 원핫으로 특성이 수십 개면 식이 길어지는데,
        그때 화면 전체가 옆으로 밀리면 안 된다 (`AppTable`과 같은 판단).
      -->
      <p class="overflow-x-auto rounded-panel bg-surface-sunken p-3 font-bold whitespace-pre">
        {{ equationText }}
      </p>
    </div>
  </section>
</template>
