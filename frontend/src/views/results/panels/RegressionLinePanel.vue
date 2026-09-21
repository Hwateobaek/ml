<script setup lang="ts">
/**
 * 회귀선 그림 (`open-decisions.md` "선형 회귀는 회귀선을 그린다").
 *
 * **계산은 없다.** 점도 선의 두 끝점도 `ml/regression-line.ts`가 만들고, 여기는 그것을
 * 캔버스에 얹는 일만 한다 (§8.3) — `LossCurvePanel`·`ClusterScatter`와 같은 자리다.
 *
 * **어느 그림인지도 여기서 안 고른다.** `kind`가 재료에 실려 오고 화면은 그 값에 따라
 * 문구와 축 이름만 고른다 — 특성 수를 세는 일이 화면에 있으면 §9.1이 막으려던 분기가
 * 여기 생긴다.
 *
 * **Chart.js 등록이 여기 있다.** 이 패널이 지연 로딩이라(`ml/metric-panels.ts`) 선형
 * 회귀를 안 돌린 학생은 차트 라이브러리를 받지 않는다.
 */

import {
  Chart,
  Legend,
  LineController,
  LineElement,
  LinearScale,
  PointElement,
  ScatterController,
  Tooltip,
  type ChartOptions,
} from 'chart.js'
import { computed, onMounted, ref, watch } from 'vue'
import { Scatter } from 'vue-chartjs'
import { useI18n } from 'vue-i18n'

import { useColumnLabels } from '@/composables/useColumnLabels'
import { useFormat } from '@/composables/useFormat'
import { columnLabel } from '@/data/column-labels'
import { scatterPointLimit } from '@/limits-switch'
import type { PanelInput } from '@/ml/metric-panels'
import { REGRESSION_TEXT, regressionChartFor } from '@/ml/regression-line'
import { theme } from '@/theme'

// **선도 그린다.** 산점도만 등록하면 `showLine`을 켠 데이터셋이 점만 남는다.
Chart.register(
  ScatterController,
  LineController,
  PointElement,
  LineElement,
  LinearScale,
  Tooltip,
  Legend,
)

const props = defineProps<{ input: PanelInput }>()

const { t } = useI18n()
const { labels: columnLabels } = useColumnLabels()
const format = useFormat()

/**
 * 그릴 것 전부. **못 세우면 `null`이고 그때 이 패널은 아무것도 안 그린다**
 * (§9.2 "없는 것을 이름으로 말하지 않는다"). 데이터를 뺀 채로 받은 파일이 그렇다.
 */
const chart = computed(() =>
  regressionChartFor(
    props.input.run.model?.format,
    props.input.modelBytes,
    props.input.dataset,
    props.input.preprocessor,
    props.input.experiment.settings,
    scatterPointLimit(),
  ),
)

/**
 * 이 그림의 문구 키들. **고르는 일은 등록부가 한다** (`REGRESSION_TEXT`) — 화면이
 * 삼항으로 고르면 두 갈래를 맞바꿔도 아무것도 안 운다.
 *
 * 그림이 없으면 아래 `v-if`가 통째로 접으므로 이 값은 안 쓰인다.
 */
const text = computed(() => REGRESSION_TEXT[chart.value?.kind ?? 'feature'])

/**
 * 배색 토큰의 실제 값. **캔버스는 CSS 클래스를 못 쓴다** — `ClusterScatter.vue`와 같은
 * 사정이고, 배색이 바뀌면 다시 읽어야 하는 것도 같다.
 *
 * **점과 선은 다른 색이다.** 같은 색이면 점이 몰린 자리에서 선이 사라진다.
 */
const dot = ref('#56b4e9')
const fit = ref('#d55e00')
const ink = ref('#475569')
const line = ref('#e2e8f0')

function readTokens(): void {
  if (typeof document === 'undefined') return
  const styles = getComputedStyle(document.documentElement)
  const token = (name: string, fallback: string): string =>
    styles.getPropertyValue(name).trim() || fallback
  dot.value = token('--color-chart-2', '#56b4e9')
  fit.value = token('--color-chart-6', '#d55e00')
  ink.value = token('--color-ink-soft', '#475569')
  line.value = token('--color-line', '#e2e8f0')
}

onMounted(readTokens)
// **`theme` ref를 본다.** DOM 속성을 게터로 읽으면 감시자가 한 번도 안 깨어난다
// (2026-08-29 전 경로 감사, `ClusterScatter.vue`의 같은 자리).
watch(theme, readTokens)

/** 축에 붙는 이름. 학생이 고쳐 부르는 이름으로 찍는다. */
const label = (name: string): string => columnLabel(name, columnLabels.value)

/** 축 이름. **키가 없는 축은 학생의 열 이름이다** (`REGRESSION_TEXT`의 머리말). */
const axisX = computed(() =>
  text.value.axisX ? t(text.value.axisX) : label(chart.value?.featureName ?? ''),
)

const axisY = computed(() =>
  text.value.axisY ? t(text.value.axisY) : label(chart.value?.targetName ?? ''),
)

/**
 * 툴팁에 적는 좌표. **Chart.js는 좌표를 `null`일 수 있는 것으로 본다** — 빈 자리를 둘 수
 * 있는 차트 종류가 있어서다. 이 그림에는 그런 점이 없지만 타입은 그것을 모른다
 * (`ClusterScatter.vue`의 같은 자리).
 */
const coordinate = (value: number | null): string =>
  value === null ? t('meta.none') : format.prediction(value)

/**
 * 점 한 벌과 선 한 벌.
 *
 * **선이 점 위에 그려져야 한다.** Chart.js는 `order`로 정렬한 뒤 **뒤에서부터** 그리므로
 * 작은 쪽이 위다 (`ml/cluster-chart.ts`가 중심점에서 같은 것을 겪었다).
 */
const chartData = computed(() => ({
  datasets: [
    {
      label: t('results.regressionData'),
      data: (chart.value?.points ?? []).map((point) => ({ x: point.x, y: point.y })),
      pointBackgroundColor: dot.value,
      pointBorderColor: dot.value,
      pointRadius: 4,
      order: 2,
    },
    {
      label: t(text.value.line),
      data: (chart.value?.line ?? []).map((point) => ({ x: point.x, y: point.y })),
      borderColor: fit.value,
      backgroundColor: fit.value,
      // **범례에도 선으로 선다.** 점으로 두면 범례에서 점 둘이 색만 다른 채로 나란히
      // 서서, 어느 쪽이 데이터이고 어느 쪽이 선인지가 색으로만 갈린다.
      pointStyle: 'line' as const,
      showLine: true,
      pointRadius: 0,
      borderWidth: 3,
      order: 1,
    },
  ],
}))

/**
 * 그림의 나머지 규칙. **애니메이션을 끈다** — 점의 상한이 그 줄에 매여 있다
 * (`limits.ts`의 `SCATTER_POINT_LIMIT`).
 */
const chartOptions = computed<ChartOptions<'scatter'>>(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  // 커서 아래에 겹친 점을 전부 세우지 않는다 — 가리킨 것 하나만 말한다.
  interaction: { mode: 'nearest' as const, intersect: true },
  scales: {
    x: {
      title: { display: true, text: axisX.value, color: ink.value },
      ticks: { color: ink.value },
      grid: { color: line.value },
      border: { color: line.value },
    },
    y: {
      title: { display: true, text: axisY.value, color: ink.value },
      ticks: { color: ink.value },
      grid: { color: line.value },
      border: { color: line.value },
    },
  },
  plugins: {
    /**
     * **범례는 그리는 차례를 따라가지 않는다** (`ml/cluster-chart.ts`의 같은 자리).
     * Chart.js가 범례 항목도 `order`로 정렬하므로, 그대로 두면 선을 위에 그리려고 준
     * 번호 때문에 **선이 데이터보다 앞에 선다.**
     */
    legend: {
      labels: {
        color: ink.value,
        usePointStyle: true,
        sort: (a, b) => (a.datasetIndex ?? 0) - (b.datasetIndex ?? 0),
      },
    },
    tooltip: {
      position: 'nearest' as const,
      callbacks: {
        label: (item) =>
          t('results.regressionPoint', {
            name: item.dataset.label ?? '',
            x: coordinate(item.parsed.x),
            y: coordinate(item.parsed.y),
          }),
      },
    },
  },
}))
</script>

<template>
  <section v-if="chart" class="flex min-w-0 flex-col gap-1.5">
    <h4 class="font-bold">
      {{ t(text.title) }}
    </h4>
    <p class="text-ink-soft">
      {{ t(text.lead) }}
    </p>

    <!--
      **높이를 상자가 쥔다.** `maintainAspectRatio: false`인 차트는 부모의 높이를 그대로
      쓰고, 안 주면 캔버스가 0px로 접힌다 (`LossCurvePanel`과 같은 자리).
    -->
    <div class="h-96 min-w-0">
      <Scatter :data="chartData" :options="chartOptions" />
    </div>

    <!--
      **표본을 뽑았으면 말한다** (#28-5). 조용히 일부만 그리면 학생은 자기 데이터가
      다 거기 있다고 믿는다.
    -->
    <p v-if="chart.drawn < chart.total" class="text-ink-faint">
      {{ t('results.regressionSample', { drawn: chart.drawn, total: chart.total }) }}
    </p>
  </section>
</template>
