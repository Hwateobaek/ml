<script setup lang="ts">
/**
 * visualize 단계. **데이터를 정하기 전에 데이터를 본다.**
 *
 * 전처리에서 학생이 고르는 것은 타깃과 특성인데, 그 판단의 재료가 화면 어디에도 없었다 —
 * 열 요약이 말하는 것은 "숫자인가·빈 칸이 몇인가·값이 몇 종류인가"까지이고, **어떤 열이
 * 타깃과 같이 움직이는지는 그림으로만 보인다.** 그래서 `data`와 `preprocess` 사이에 선다
 * (`router/steps.ts`의 `STEP_IDS`).
 *
 * **계산은 여기 없다.** 구간·분위수·상관은 `data/visualize.ts`의 순수 함수이고
 * (`tests/visualize.spec.ts`가 지킨다) 여기는 그것을 Chart.js와 표에 얹는다 —
 * `ClusterScatter.vue`가 `ml/cluster-chart.ts`와 갈라지는 것과 같은 자리다.
 *
 * **배색 토큰을 CSS에서 읽는다.** 그 방법과 감시자의 이유는 `ClusterScatter.vue`의
 * `readTokens` 머리말에 있다 — 같은 것을 두 번 적지 않는다.
 */

import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  ScatterController,
  Tooltip,
  type PointStyle,
  type TooltipItem,
} from 'chart.js'
import { computed, onMounted, ref, watch } from 'vue'
import { Bar, Line, Scatter } from 'vue-chartjs'
import { useI18n } from 'vue-i18n'

import AppBadge from '@/components/AppBadge.vue'
import AppCard from '@/components/AppCard.vue'
import AppEmpty from '@/components/AppEmpty.vue'
import StepHeader from '@/components/StepHeader.vue'
import { useColumnLabels } from '@/composables/useColumnLabels'
import { useFormat } from '@/composables/useFormat'
import { summarizeColumns } from '@/data/columns'
import {
  correlationMatrix,
  boxPlot,
  categoryCounts,
  crossTab,
  defaultLineAxis,
  groupedBoxPlots,
  histogram,
  lineChart,
  numericPairs,
  numericValues,
} from '@/data/visualize'
import { FALLBACK_PALETTE, POINT_SHAPES } from '@/ml/cluster-chart'
import { readDataset } from '@/project/dataset'
import { useProjectStore } from '@/stores/project'
import { theme } from '@/theme'

Chart.register(
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  ScatterController,
  Tooltip,
  Legend,
)

const { t } = useI18n()
const format = useFormat()
const project = useProjectStore()

/**
 * 화면에 찍는 이름. **고르는 값은 원본 이름 그대로다** — 축과 상관 행렬이 전부 원본
 * 이름으로 열을 찾으므로, 별칭을 값으로 쓰면 이름을 고친 열이 사라진다.
 */
const { label } = useColumnLabels()

/** 정본을 파싱한 표. 바이트가 같으면 다시 파싱하지 않는다 (project/dataset.ts). */
const dataset = computed(() => readDataset(project.file))

const summaries = computed(() => (dataset.value ? summarizeColumns(dataset.value) : []))

/** 그릴 수 있는 열. **수치 열만이다** — 산점도의 축과 상관계수가 둘 다 수를 요구한다. */
const numericColumns = computed(() =>
  summaries.value.filter((column) => column.kind === 'numeric').map((column) => column.name),
)

/**
 * 범주로 볼 수 있는 열 — 산점도의 색, 막대그래프, 그룹별 상자그림, 교차표가 함께 쓴다.
 * **값 종류가 팔레트보다 많으면 뺀다** — 색이 돌아가기 시작하면 서로 다른 범주가 같은 색이
 * 되고, 그림이 거짓말을 한다. 막대와 교차표도 그 이상이면 칸이 읽히지 않는다.
 *
 * **값 종류가 적은 수치 열도 든다.** `Survived`의 `0/1`, `Pclass`의 `1/2/3`처럼 범주를
 * 숫자로 적은 열이 교실 데이터에 흔하고, 그 열이 바로 분류의 타깃이다. **수치 열은 색
 * 한 바퀴(모양이 안 바뀌는 만큼)까지만이다** — 스무 행짜리 표에서는 길이·무게 같은 연속
 * 값도 종류가 스물 남짓이라, 범주형과 같은 기준이면 목록이 연속 열로 찬다.
 */
const groupColumns = computed(() =>
  summaries.value
    .filter((column) => column.unique >= 2)
    .filter((column) =>
      column.kind === 'categorical'
        ? column.unique <= FALLBACK_PALETTE.length * POINT_SHAPES.length
        : column.unique <= FALLBACK_PALETTE.length,
    )
    .map((column) => column.name),
)

/** 색으로 나누지 않는다는 선택. **빈 문자열이 그 뜻이다** — 열 이름은 비어 있을 수 없다. */
const NO_GROUP = ''

const distributionColumn = ref('')
const xColumn = ref('')
const yColumn = ref('')
const groupColumn = ref(NO_GROUP)
/** 막대그래프의 열. */
const countColumn = ref('')
/** 그룹별 상자그림 — 값을 볼 수치 열과 나눌 범주 열. */
const boxValueColumn = ref('')
const boxGroupColumn = ref('')
/** 교차표의 행과 열. */
const crossRowColumn = ref('')
const crossColumnColumn = ref('')
/** 선그래프의 가로축 열과 선으로 그릴 열들. */
const lineXColumn = ref('')
const lineYColumns = ref<string[]>([])

/** 선그래프의 가로축이 될 수 있는 열. **모든 열이다** — 연도는 숫자로도, 글자로도 적힌다. */
const allColumns = computed(() => summaries.value.map((column) => column.name))

/** 선으로 그릴 수 있는 열. 가로축으로 쓰는 열은 뺀다 — 자기 자신과의 선은 대각선뿐이다. */
const lineYOptions = computed(() =>
  numericColumns.value.filter((name) => name !== lineXColumn.value),
)

watch(
  allColumns,
  (columns) => {
    if (!columns.includes(lineXColumn.value)) {
      lineXColumn.value = defaultLineAxis(columns)
    }
  },
  { immediate: true },
)

watch(
  lineYOptions,
  (options) => {
    const kept = lineYColumns.value.filter((name) => options.includes(name))
    lineYColumns.value = kept.length > 0 ? kept : options.slice(0, 1)
  },
  { immediate: true },
)

function toggleLineColumn(name: string, checked: boolean): void {
  // 열 차례를 표의 차례로 유지한다 — 누른 차례로 쌓으면 선의 색이 누를 때마다 바뀐다.
  const next = new Set(lineYColumns.value)
  if (checked) next.add(name)
  else next.delete(name)
  lineYColumns.value = lineYOptions.value.filter((option) => next.has(option))
}

/**
 * 열이 바뀌면 고른 값을 다시 맞춘다.
 *
 * **없는 열 이름이 남아 있으면 그림이 빈 채로 선다.** 프로젝트를 새로 열거나 데이터를
 * 바꾸면 그 자리가 실제로 생긴다.
 */
watch(
  numericColumns,
  (columns) => {
    if (!columns.includes(distributionColumn.value)) distributionColumn.value = columns[0] ?? ''
    if (!columns.includes(xColumn.value)) xColumn.value = columns[0] ?? ''
    if (!columns.includes(yColumn.value)) yColumn.value = columns[1] ?? columns[0] ?? ''
    if (!columns.includes(boxValueColumn.value)) boxValueColumn.value = columns[0] ?? ''
  },
  { immediate: true },
)

watch(
  groupColumns,
  (columns) => {
    if (groupColumn.value !== NO_GROUP && !columns.includes(groupColumn.value)) {
      groupColumn.value = NO_GROUP
    }
    // 수치 열이 범주로도 들 수 있으니, 처음 고르는 값은 **범주형 열을 먼저** 잡는다.
    const categorical = new Set(
      summaries.value
        .filter((column) => column.kind === 'categorical')
        .map((column) => column.name),
    )
    const ordered = [
      ...columns.filter((name) => categorical.has(name)),
      ...columns.filter((name) => !categorical.has(name)),
    ]
    if (!columns.includes(countColumn.value)) countColumn.value = ordered[0] ?? ''
    if (!columns.includes(boxGroupColumn.value)) boxGroupColumn.value = ordered[0] ?? ''
    if (!columns.includes(crossRowColumn.value)) crossRowColumn.value = ordered[0] ?? ''
    if (!columns.includes(crossColumnColumn.value)) {
      crossColumnColumn.value = ordered[1] ?? ordered[0] ?? ''
    }
  },
  { immediate: true },
)

/* ── 배색 토큰 (`ClusterScatter.vue`의 `readTokens`와 같은 방법) ── */

const palette = ref<readonly string[]>(FALLBACK_PALETTE)
const surface = ref('#ffffff')
const ink = ref('#475569')
const line = ref('#e2e8f0')

function readTokens(): void {
  if (typeof document === 'undefined') return
  const styles = getComputedStyle(document.documentElement)
  const token = (name: string, fallback: string): string =>
    styles.getPropertyValue(name).trim() || fallback

  palette.value = FALLBACK_PALETTE.map((fallback, index) =>
    token(`--color-chart-${index + 1}`, fallback),
  )
  surface.value = token('--color-surface', '#ffffff')
  ink.value = token('--color-ink-soft', '#475569')
  line.value = token('--color-line', '#e2e8f0')
}

onMounted(readTokens)
watch(theme, readTokens)

/** 범주 하나의 색. 팔레트를 돌려 쓰고, 한 바퀴 돌면 모양이 바뀐다 (`clusterColor`와 같은 규칙). */
function groupColor(index: number): string {
  const colors = palette.value
  return colors[index % colors.length] ?? ink.value
}

function groupShape(index: number): PointStyle {
  const count = palette.value.length || 1
  return POINT_SHAPES[Math.floor(index / count) % POINT_SHAPES.length] ?? 'circle'
}

/* ── 분포 ── */

const distributionValues = computed(() =>
  dataset.value && distributionColumn.value
    ? numericValues(dataset.value, distributionColumn.value)
    : [],
)

const bins = computed(() => histogram(distributionValues.value))

/**
 * 다섯 수 요약과 상자그림 (`data/visualize.ts`의 `boxPlot`). **경계는 전처리기가 자르는
 * 것과 같은 함수에서 나온다** — 다만 여기는 전체 데이터이고 전처리기는 훈련 데이터라,
 * 두 경계의 숫자는 조금 다를 수 있다 (open-decisions.md "이상치는 훈련 데이터의 IQR로
 * 클리핑한다").
 */
const summary = computed(() => boxPlot(distributionValues.value))

/** 상자그림의 가로 폭. 좌표는 이 폭 안의 값이고, 그림은 `preserveAspectRatio="none"`으로 늘어난다. */
const BOX_WIDTH = 1000

/**
 * 값 → 상자그림의 가로 자리. **최솟값에서 최댓값까지를 폭 전체로 편다.**
 *
 * 값이 전부 같으면 폭이 0이라 가운데에 둔다 — 나눗셈이 무한대가 되면 SVG가 아무것도
 * 안 그린다.
 */
function boxX(value: number): number {
  const plot = summary.value
  if (!plot || plot.max === plot.min) return BOX_WIDTH / 2
  return ((value - plot.min) / (plot.max - plot.min)) * BOX_WIDTH
}

const histogramData = computed(() => ({
  labels: bins.value.map((bin) => format.stat(bin.start)),
  datasets: [
    {
      data: bins.value.map((bin) => bin.count),
      backgroundColor: groupColor(1),
      borderRadius: 2,
    },
  ],
}))

const histogramOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        title: (items: TooltipItem<'bar'>[]) => {
          const bin = bins.value[items[0]?.dataIndex ?? 0]
          if (bin === undefined) return ''
          return t('visualize.binRange', {
            start: format.stat(bin.start),
            end: format.stat(bin.end),
          })
        },
        // 수를 그대로 넘겨야 복수형이 골라진다 (`TabularPrepSummary.vue`의 `rows`와 같은 꼴).
        label: (item: TooltipItem<'bar'>) => t('visualize.binCount', item.parsed.y ?? 0),
      },
    },
  },
  scales: {
    x: { grid: { color: line.value }, ticks: { color: ink.value } },
    y: { beginAtZero: true, grid: { color: line.value }, ticks: { color: ink.value } },
  },
}))

/* ── 산점도 ── */

/** 범주별로 나눈 점들. 색으로 안 나누면 묶음이 하나다. */
const scatterGroups = computed(() => {
  const table = dataset.value
  if (table === null || xColumn.value === '' || yColumn.value === '') return []
  const pairs = numericPairs(table, xColumn.value, yColumn.value)

  if (groupColumn.value === NO_GROUP) {
    return [{ name: t('visualize.allRows'), points: pairs }]
  }

  const index = table.columns.indexOf(groupColumn.value)
  const buckets = new Map<string, { x: number; y: number }[]>()
  for (const pair of pairs) {
    // **`label`이라고 부르지 않는다** — 위 `useColumnLabels`의 것을 가린다. 여기 담기는
    // 것은 열 이름이 아니라 그 행의 범주 값이라 애초에 다른 물건이다.
    const category = table.rows[pair.row]?.[index] ?? ''
    const bucket = buckets.get(category)
    if (bucket === undefined) buckets.set(category, [pair])
    else bucket.push(pair)
  }
  return [...buckets.entries()].map(([name, points]) => ({ name, points }))
})

const scatterData = computed(() => ({
  datasets: scatterGroups.value.map((group, index) => ({
    label: group.name,
    data: group.points.map((point) => ({ x: point.x, y: point.y })),
    backgroundColor: groupColor(index),
    borderColor: surface.value,
    borderWidth: 1,
    pointStyle: groupShape(index),
    radius: 4,
    hoverRadius: 6,
  })),
}))

const scatterOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: {
    // 묶음이 하나뿐이면 범례가 아무것도 안 가른다 — 이름 하나를 띄우려고 자리를 쓰지 않는다.
    legend: {
      display: groupColumn.value !== NO_GROUP,
      labels: { color: ink.value, usePointStyle: true },
    },
    tooltip: {
      callbacks: {
        label: (item: TooltipItem<'scatter'>) =>
          t('visualize.point', {
            name: item.dataset.label ?? '',
            x: format.stat(item.parsed.x ?? 0),
            y: format.stat(item.parsed.y ?? 0),
          }),
      },
    },
  },
  scales: {
    x: {
      type: 'linear' as const,
      title: { display: true, text: label(xColumn.value), color: ink.value },
      grid: { color: line.value },
      ticks: { color: ink.value },
    },
    y: {
      title: { display: true, text: label(yColumn.value), color: ink.value },
      grid: { color: line.value },
      ticks: { color: ink.value },
    },
  },
}))

/* ── 상관 히트맵 ── */

const correlations = computed(() =>
  dataset.value ? correlationMatrix(dataset.value, numericColumns.value) : [],
)

/**
 * 상관 한 칸의 배경.
 *
 * **두 색은 팔레트에서 온다** — 새 색을 지어내지 않는다. 진하기가 세기이고 0은 투명이라
 * 바탕색이 그대로 중간값이 된다. 색맹 검사를 통과한 주황·하늘 짝이라 붉음/푸름보다 낫다.
 */
function correlationStyle(value: number): Record<string, string> {
  const strength = Math.min(Math.abs(value), 1) * 0.85
  const hex = value < 0 ? groupColor(1) : groupColor(0)
  return { backgroundColor: withAlpha(hex, strength) }
}

/** `#rrggbb`에 투명도를 붙인다. 토큰이 다른 표기로 오면 그대로 돌려준다. */
function withAlpha(hex: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(hex.trim())
  if (match === null) return hex
  const value = match[1] ?? ''
  const red = Number.parseInt(value.slice(0, 2), 16)
  const green = Number.parseInt(value.slice(2, 4), 16)
  const blue = Number.parseInt(value.slice(4, 6), 16)
  return `rgba(${red}, ${green}, ${blue}, ${alpha.toFixed(3)})`
}

/* ── 범주 막대그래프 ── */

const counts = computed(() =>
  dataset.value && countColumn.value ? categoryCounts(dataset.value, countColumn.value) : [],
)

/** 빈 칸을 뺀 행 수. 툴팁의 비율이 이것을 분모로 쓴다 — 막대 높이의 합과 같아야 한다. */
const countTotal = computed(() => counts.value.reduce((sum, item) => sum + item.count, 0))

/** 막대에 들지 못한 빈 칸의 수. 0이 아니면 그래프 아래에 말한다. */
const countBlank = computed(() => (dataset.value?.rows.length ?? 0) - countTotal.value)

const countData = computed(() => ({
  labels: counts.value.map((item) => item.value),
  datasets: [
    {
      data: counts.value.map((item) => item.count),
      backgroundColor: counts.value.map((_, index) => groupColor(index)),
      borderRadius: 2,
    },
  ],
}))

const countOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  plugins: {
    legend: { display: false },
    tooltip: {
      callbacks: {
        label: (item: TooltipItem<'bar'>) =>
          t('visualize.countValue', {
            count: item.parsed.y ?? 0,
            percent: format.percent(
              countTotal.value === 0 ? 0 : (item.parsed.y ?? 0) / countTotal.value,
            ),
          }),
      },
    },
  },
  scales: {
    x: { grid: { color: line.value }, ticks: { color: ink.value } },
    y: {
      beginAtZero: true,
      grid: { color: line.value },
      ticks: { color: ink.value, precision: 0 },
    },
  },
}))

/* ── 선그래프 ── */

const lines = computed(() =>
  dataset.value && lineXColumn.value
    ? lineChart(dataset.value, lineXColumn.value, lineYColumns.value)
    : { labels: [], series: [], averaged: false },
)

/**
 * 선마다 색과 점 모양이 **둘 다** 다르다. 교과서 그림처럼 흑백으로 인쇄해도 선이 갈려야
 * 하고, 색을 못 가리는 학생에게도 그렇다.
 */
const LINE_POINTS: readonly PointStyle[] = [
  'circle',
  'rect',
  'triangle',
  'star',
  'rectRot',
  'crossRot',
]

const lineData = computed(() => ({
  labels: [...lines.value.labels],
  datasets: lines.value.series.map((series, index) => ({
    label: label(series.column),
    data: [...series.values],
    borderColor: groupColor(index),
    backgroundColor: groupColor(index),
    pointStyle: LINE_POINTS[index % LINE_POINTS.length] ?? 'circle',
    pointRadius: 3,
    pointHoverRadius: 6,
    borderWidth: 2.5,
    tension: 0,
    // 빈 자리는 이어 긋지 않는다 — 없던 값을 선이 지어낸 것처럼 보인다.
    spanGaps: false,
  })),
}))

const lineOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false as const,
  interaction: { mode: 'index' as const, intersect: false },
  plugins: {
    legend: { display: true, labels: { color: ink.value, usePointStyle: true } },
    tooltip: {
      callbacks: {
        label: (item: TooltipItem<'line'>) =>
          t('visualize.lineValue', {
            name: item.dataset.label ?? '',
            value: format.stat(item.parsed.y ?? 0),
          }),
      },
    },
  },
  scales: {
    x: {
      title: { display: true, text: label(lineXColumn.value), color: ink.value },
      grid: { color: line.value },
      ticks: { color: ink.value },
    },
    y: { grid: { color: line.value }, ticks: { color: ink.value } },
  },
}))

/* ── 그룹별 상자그림 ── */

const groupedPlots = computed(() =>
  dataset.value && boxValueColumn.value && boxGroupColumn.value
    ? groupedBoxPlots(dataset.value, boxValueColumn.value, boxGroupColumn.value)
    : [],
)

/**
 * 모든 상자가 함께 쓰는 가로 눈금. **범주마다 따로 펴면 안 된다** — 그러면 상자의 자리가
 * 전부 같아 보여서, 이 그림의 목적인 "범주끼리 견주기"가 사라진다.
 */
const groupedScale = computed(() => {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  for (const { plot } of groupedPlots.value) {
    if (plot.min < min) min = plot.min
    if (plot.max > max) max = plot.max
  }
  return { min, max }
})

function groupedX(value: number): number {
  const { min, max } = groupedScale.value
  if (!(max > min)) return BOX_WIDTH / 2
  return ((value - min) / (max - min)) * BOX_WIDTH
}

/* ── 교차표 ── */

const cross = computed(() =>
  dataset.value && crossRowColumn.value && crossColumnColumn.value
    ? crossTab(dataset.value, crossRowColumn.value, crossColumnColumn.value)
    : { rows: [], columns: [], counts: [] },
)

const crossRowTotals = computed(() =>
  cross.value.counts.map((line) => line.reduce((sum, count) => sum + count, 0)),
)

const crossColumnTotals = computed(() =>
  cross.value.columns.map((_, index) =>
    cross.value.counts.reduce((sum, line) => sum + (line[index] ?? 0), 0),
  ),
)

const crossTotal = computed(() => crossRowTotals.value.reduce((sum, count) => sum + count, 0))

/** 가장 많은 칸. 진하기의 기준이다. */
const crossMax = computed(() => {
  let max = 0
  for (const line of cross.value.counts) for (const count of line) if (count > max) max = count
  return max
})

/** 교차표 한 칸의 배경. **색은 하나다** — 개수에는 방향이 없어서 상관처럼 두 색을 안 쓴다. */
function crossStyle(count: number): Record<string, string> {
  const strength = crossMax.value === 0 ? 0 : (count / crossMax.value) * 0.85
  return { backgroundColor: withAlpha(groupColor(1), strength) }
}
</script>

<template>
  <!--
    **다른 단계 화면과 같은 뼈대다** (`views/data/TabularPanel.vue`의 머리말에 `min-h-full`이
    `h-full`이 아닌 이유가 적혀 있다).

    **여백과 최소 높이를 빼먹으면 스크롤이 바깥으로 샌다** (2026-09-10, 사용자 화면).
    작업 공간(`AppShell`의 `<main>`)이 `md:overflow-auto`로 **자기 안에서** 굴러가는데,
    그 안을 채우지 않으면 넘치는 것이 문서 쪽이 되어 **도구 막대와 레일까지 화면 위로
    밀려 올라간다** — 다른 탭에서는 안 그러니 이 탭만 고장 난 것으로 보인다.
  -->
  <div class="flex min-h-full flex-col gap-5 p-4 sm:p-5">
    <StepHeader :title="t('steps.visualize.label')" :purpose="t('steps.visualize.purpose')" />

    <AppEmpty
      v-if="dataset === null"
      :reason="t('visualize.emptyReason')"
      :next="t('visualize.emptyNext')"
    />

    <!--
      **수치 열이 없으면 그릴 것이 없다.** 이유를 말하는 자리를 비워 두면 화면이 통째로
      빈 채로 서고, 교실에서는 그것이 고장으로 읽힌다 (`ClusterScatter.vue`의 같은 자리).
    -->
    <!--
      **범주 그림은 수치 열이 없어도 선다** (막대그래프·교차표). 그래서 수치 열이 없다는
      안내는 화면을 통째로 대신하지 않고, 수치 그림의 자리에만 선다.
    -->
    <template v-else>
      <AppEmpty
        v-if="numericColumns.length === 0"
        :reason="t('visualize.noNumericReason')"
        :next="t('visualize.noNumericNext')"
      />

      <template v-else>
        <!-- 분포 -->
        <!--
        **`min-w-0`이 세 카드에 다 있다.** 세로 flex의 칸은 기본이 `min-width: auto`라
        **안쪽이 넓으면 칸이 그만큼 벌어진다** — 상관 표는 열 수만큼 넓어지므로, 이게
        없으면 카드가 작업 공간을 밀어 화면이 옆으로 넘친다. 넘치는 폭은 그 표가
        자기 안에서 굴려 보여준다(`overflow-x-auto`).
      -->
        <AppCard
          class="min-w-0"
          :title="t('visualize.distribution')"
          :description="t('visualize.distributionLead')"
        >
          <div class="flex flex-col gap-4">
            <label class="flex flex-wrap items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.column') }}</span>
              <select
                v-model="distributionColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in numericColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>

            <div class="h-80 min-w-0">
              <Bar :data="histogramData" :options="histogramOptions" />
            </div>

            <dl v-if="summary" class="flex flex-wrap gap-x-6 gap-y-1">
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.min') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ format.stat(summary.min) }}</dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.q1') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ format.stat(summary.q1) }}</dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.median') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ format.stat(summary.median) }}</dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.q3') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ format.stat(summary.q3) }}</dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.max') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ format.stat(summary.max) }}</dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.outlierBounds') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">
                  {{
                    summary.bounds
                      ? t('visualize.outlierBoundsValue', {
                          low: format.stat(summary.bounds.low),
                          high: format.stat(summary.bounds.high),
                        })
                      : t('visualize.outlierBoundsFlat')
                  }}
                </dd>
              </div>
              <div class="flex items-baseline gap-1.5">
                <dt>
                  <AppBadge>{{ t('visualize.outlierCount') }}</AppBadge>
                </dt>
                <dd class="font-bold tabular-nums text-ink">{{ summary.outlierCount }}</dd>
              </div>
            </dl>

            <!--
            **상자그림은 히스토그램 아래에 선다.** 같은 열의 두 그림이라 따로 고르지 않는다.

            **SVG를 직접 그린다.** Chart.js에는 상자그림이 없고, 그것 하나 때문에 플러그인을
            들이지 않는다 (상관 표를 표로 그린 것과 같은 판단). 늘어나는 그림이라 선은
            `vector-effect`로 굵기를 지키고, 글자는 그림 안에 두지 않는다 — 숫자는 위
            배지가 말한다.
          -->
            <div v-if="summary" class="flex flex-col gap-1.5">
              <h4 class="font-bold">{{ t('visualize.boxPlot') }}</h4>
              <p class="text-ink-soft">{{ t('visualize.boxPlotLead') }}</p>
              <svg
                class="h-16 w-full"
                :viewBox="`0 0 ${BOX_WIDTH} 60`"
                preserveAspectRatio="none"
                role="img"
                :aria-label="t('visualize.boxPlot')"
              >
                <line
                  :x1="boxX(summary.whiskerLow)"
                  :x2="boxX(summary.q1)"
                  y1="30"
                  y2="30"
                  :stroke="ink"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  :x1="boxX(summary.q3)"
                  :x2="boxX(summary.whiskerHigh)"
                  y1="30"
                  y2="30"
                  :stroke="ink"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  v-for="(end, index) in [summary.whiskerLow, summary.whiskerHigh]"
                  :key="index"
                  :x1="boxX(end)"
                  :x2="boxX(end)"
                  y1="18"
                  y2="42"
                  :stroke="ink"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <rect
                  :x="boxX(summary.q1)"
                  y="10"
                  :width="Math.max(boxX(summary.q3) - boxX(summary.q1), 1)"
                  height="40"
                  :fill="groupColor(1)"
                  fill-opacity="0.35"
                  :stroke="groupColor(1)"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  :x1="boxX(summary.median)"
                  :x2="boxX(summary.median)"
                  y1="10"
                  y2="50"
                  :stroke="ink"
                  stroke-width="2.5"
                  vector-effect="non-scaling-stroke"
                />
                <!--
                **점이 아니라 짧은 세로선이다.** 가로로만 늘어나는 그림이라 원은 타원이
                된다. 색은 상자와 다른 팔레트 칸이라 겹쳐도 갈린다.
              -->
                <line
                  v-for="value in summary.marks"
                  :key="value"
                  :x1="boxX(value)"
                  :x2="boxX(value)"
                  y1="20"
                  y2="40"
                  :stroke="groupColor(0)"
                  stroke-width="2"
                  vector-effect="non-scaling-stroke"
                />
              </svg>
            </div>
          </div>
        </AppCard>
      </template>

      <!-- 범주 막대그래프 -->
      <AppCard
        v-if="groupColumns.length > 0"
        class="min-w-0"
        :title="t('visualize.counts')"
        :description="t('visualize.countsLead')"
      >
        <div class="flex flex-col gap-4">
          <label class="flex flex-wrap items-center gap-2">
            <span class="font-bold text-ink-soft">{{ t('visualize.column') }}</span>
            <select
              v-model="countColumn"
              class="rounded-field border border-line-strong bg-surface px-2 py-1"
            >
              <option v-for="name in groupColumns" :key="name" :value="name">
                {{ label(name) }}
              </option>
            </select>
          </label>

          <div class="h-80 min-w-0">
            <Bar :data="countData" :options="countOptions" />
          </div>

          <p v-if="countBlank > 0" class="text-ink-faint">
            {{ t('visualize.blankExcluded', { count: countBlank }) }}
          </p>
        </div>
      </AppCard>

      <!-- 그룹별 상자그림 -->
      <AppCard
        v-if="numericColumns.length > 0 && groupColumns.length > 0"
        class="min-w-0"
        :title="t('visualize.groupedBoxPlot')"
        :description="t('visualize.groupedBoxPlotLead')"
      >
        <div class="flex flex-col gap-4">
          <div class="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-x-5">
            <label class="flex items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.valueColumn') }}</span>
              <select
                v-model="boxValueColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in numericColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>

            <label class="flex items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.groupColumn') }}</span>
              <select
                v-model="boxGroupColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in groupColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>
          </div>

          <!--
            **범주마다 한 줄이고 가로 눈금은 모두 같다** (`groupedScale`). 이름과 숫자는 그림
            밖 HTML이다 — 가로로만 늘어나는 SVG 안에 글자를 두면 찌그러진다.
          -->
          <div class="flex flex-col gap-2">
            <div
              v-for="(item, index) in groupedPlots"
              :key="item.group"
              class="grid grid-cols-1 items-center gap-1 sm:grid-cols-4 sm:gap-3"
            >
              <div class="min-w-0">
                <p class="truncate font-bold" :title="item.group">{{ item.group }}</p>
                <p class="text-ink-soft tabular-nums">
                  {{
                    t('visualize.groupSummary', {
                      median: format.stat(item.plot.median),
                      outliers: item.plot.outlierCount,
                    })
                  }}
                </p>
              </div>
              <svg
                class="h-12 w-full sm:col-span-3"
                :viewBox="`0 0 ${BOX_WIDTH} 60`"
                preserveAspectRatio="none"
                role="img"
                :aria-label="
                  t('visualize.groupBoxLabel', {
                    group: item.group,
                    min: format.stat(item.plot.min),
                    median: format.stat(item.plot.median),
                    max: format.stat(item.plot.max),
                  })
                "
              >
                <line
                  :x1="groupedX(item.plot.whiskerLow)"
                  :x2="groupedX(item.plot.whiskerHigh)"
                  y1="30"
                  y2="30"
                  :stroke="ink"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  v-for="(end, endIndex) in [item.plot.whiskerLow, item.plot.whiskerHigh]"
                  :key="endIndex"
                  :x1="groupedX(end)"
                  :x2="groupedX(end)"
                  y1="18"
                  y2="42"
                  :stroke="ink"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <rect
                  :x="groupedX(item.plot.q1)"
                  y="10"
                  :width="Math.max(groupedX(item.plot.q3) - groupedX(item.plot.q1), 1)"
                  height="40"
                  :fill="groupColor(index)"
                  fill-opacity="0.45"
                  :stroke="groupColor(index)"
                  stroke-width="1.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  :x1="groupedX(item.plot.median)"
                  :x2="groupedX(item.plot.median)"
                  y1="10"
                  y2="50"
                  :stroke="ink"
                  stroke-width="2.5"
                  vector-effect="non-scaling-stroke"
                />
                <line
                  v-for="value in item.plot.marks"
                  :key="value"
                  :x1="groupedX(value)"
                  :x2="groupedX(value)"
                  y1="20"
                  y2="40"
                  :stroke="ink"
                  stroke-width="2"
                  vector-effect="non-scaling-stroke"
                />
              </svg>
            </div>

            <!-- 공통 눈금의 양 끝. 상자들이 어디쯤인지 숫자로 읽는 자리다. -->
            <div
              v-if="groupedPlots.length > 0"
              class="grid grid-cols-1 gap-1 sm:grid-cols-4 sm:gap-3"
            >
              <div class="hidden sm:block"></div>
              <div class="flex justify-between text-ink-soft tabular-nums sm:col-span-3">
                <span>{{ format.stat(groupedScale.min) }}</span>
                <span>{{ format.stat(groupedScale.max) }}</span>
              </div>
            </div>
          </div>
        </div>
      </AppCard>

      <template v-if="numericColumns.length > 0">
        <!-- 산점도 -->
        <AppCard
          class="min-w-0"
          :title="t('visualize.scatter')"
          :description="t('visualize.scatterLead')"
        >
          <div v-if="numericColumns.length < 2" class="text-ink-soft">
            {{ t('visualize.scatterNeedsTwo', { count: numericColumns.length }) }}
          </div>

          <div v-else class="flex flex-col gap-4">
            <div class="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-x-5">
              <label class="flex items-center gap-2">
                <span class="font-bold text-ink-soft">{{ t('visualize.axisX') }}</span>
                <select
                  v-model="xColumn"
                  class="rounded-field border border-line-strong bg-surface px-2 py-1"
                >
                  <option v-for="name in numericColumns" :key="name" :value="name">
                    {{ label(name) }}
                  </option>
                </select>
              </label>

              <label class="flex items-center gap-2">
                <span class="font-bold text-ink-soft">{{ t('visualize.axisY') }}</span>
                <select
                  v-model="yColumn"
                  class="rounded-field border border-line-strong bg-surface px-2 py-1"
                >
                  <option v-for="name in numericColumns" :key="name" :value="name">
                    {{ label(name) }}
                  </option>
                </select>
              </label>

              <label v-if="groupColumns.length > 0" class="flex items-center gap-2">
                <span class="font-bold text-ink-soft">{{ t('visualize.groupBy') }}</span>
                <select
                  v-model="groupColumn"
                  class="rounded-field border border-line-strong bg-surface px-2 py-1"
                >
                  <option :value="NO_GROUP">{{ t('visualize.groupNone') }}</option>
                  <option v-for="name in groupColumns" :key="name" :value="name">
                    {{ label(name) }}
                  </option>
                </select>
              </label>
            </div>

            <div class="h-96 min-w-0">
              <Scatter :data="scatterData" :options="scatterOptions" />
            </div>
          </div>
        </AppCard>

        <!-- 선그래프 -->
        <AppCard
          class="min-w-0"
          :title="t('visualize.line')"
          :description="t('visualize.lineLead')"
        >
          <div class="flex flex-col gap-4">
            <label class="flex flex-wrap items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.axisX') }}</span>
              <select
                v-model="lineXColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in allColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>

            <fieldset class="flex flex-wrap items-center gap-x-4 gap-y-2">
              <legend class="mb-2 font-bold text-ink-soft">{{ t('visualize.lineColumns') }}</legend>
              <label
                v-for="name in lineYOptions"
                :key="name"
                class="flex cursor-pointer items-center gap-2"
              >
                <input
                  type="checkbox"
                  class="size-4 accent-brand"
                  :checked="lineYColumns.includes(name)"
                  @change="toggleLineColumn(name, ($event.target as HTMLInputElement).checked)"
                />
                <span>{{ label(name) }}</span>
              </label>
            </fieldset>

            <p v-if="lineYColumns.length === 0" class="text-ink-soft">
              {{ t('visualize.lineNeedsColumn') }}
            </p>
            <div v-else class="h-96 min-w-0">
              <Line :data="lineData" :options="lineOptions" />
            </div>

            <p v-if="lines.averaged && lineYColumns.length > 0" class="text-ink-faint">
              {{ t('visualize.lineAveraged', { column: label(lineXColumn) }) }}
            </p>
          </div>
        </AppCard>

        <!-- 상관 히트맵 -->
        <AppCard
          class="min-w-0"
          :title="t('visualize.correlation')"
          :description="t('visualize.correlationLead')"
        >
          <div v-if="numericColumns.length < 2" class="text-ink-soft">
            {{ t('visualize.scatterNeedsTwo', { count: numericColumns.length }) }}
          </div>

          <!--
          **표로 그린다.** 히트맵을 위해 차트 라이브러리를 하나 더 들이지 않는다 - 값이
          칸에 그대로 적히므로 색을 못 읽어도 읽히고, 화면 낭독기도 이 표를 읽는다.
        -->
          <div v-else class="overflow-x-auto">
            <!--
            **표의 이름은 `aria-label`로 준다. `<caption class="sr-only">`는 쓰면 안 된다**
            (2026-09-10, 사용자가 화면에서 잡았다).

            `sr-only`는 `position: absolute`인데, **`<caption>`에 걸면 컨테이닝 블록이
            표를 벗어나 문서 최상위에 붙는다.** 그러면 셸의 `overflow: hidden`이 그것을
            못 자르고, 그 한 줄이 **문서를 1,595px까지 늘려** 작업 공간째 화면 밖으로
            밀어 올렸다 — 화면에는 아무것도 안 보이는 자리라 원인을 짚기 어렵다.
          -->
            <table class="border-collapse" :aria-label="t('visualize.correlation')">
              <thead>
                <tr>
                  <td></td>
                  <th
                    v-for="name in numericColumns"
                    :key="name"
                    scope="col"
                    class="px-2 py-1 text-left font-bold text-ink-soft"
                  >
                    {{ label(name) }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(name, row) in numericColumns" :key="name">
                  <th
                    scope="row"
                    class="py-1 pr-3 text-left font-bold whitespace-nowrap text-ink-soft"
                  >
                    {{ label(name) }}
                  </th>
                  <td
                    v-for="(other, cell) in numericColumns"
                    :key="other"
                    class="px-2 py-1 text-center tabular-nums"
                    :style="correlationStyle(correlations[row]?.[cell] ?? 0)"
                  >
                    {{ format.metric(correlations[row]?.[cell] ?? 0, 'number') }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </AppCard>
      </template>

      <!-- 교차표 -->
      <AppCard
        v-if="groupColumns.length > 0"
        class="min-w-0"
        :title="t('visualize.crossTab')"
        :description="t('visualize.crossTabLead')"
      >
        <div class="flex flex-col gap-4">
          <div class="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:gap-x-5">
            <label class="flex items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.crossRows') }}</span>
              <select
                v-model="crossRowColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in groupColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>

            <label class="flex items-center gap-2">
              <span class="font-bold text-ink-soft">{{ t('visualize.crossColumns') }}</span>
              <select
                v-model="crossColumnColumn"
                class="rounded-field border border-line-strong bg-surface px-2 py-1"
              >
                <option v-for="name in groupColumns" :key="name" :value="name">
                  {{ label(name) }}
                </option>
              </select>
            </label>
          </div>

          <!-- 상관 표와 같은 이유로 `aria-label`이다 (`<caption class="sr-only">`를 쓰지 마라). -->
          <div class="overflow-x-auto">
            <table class="border-collapse" :aria-label="t('visualize.crossTab')">
              <thead>
                <tr>
                  <th
                    scope="col"
                    class="py-1 pr-3 text-left font-normal whitespace-nowrap text-ink-faint"
                  >
                    {{
                      t('visualize.crossCorner', {
                        rows: label(crossRowColumn),
                        columns: label(crossColumnColumn),
                      })
                    }}
                  </th>
                  <th
                    v-for="name in cross.columns"
                    :key="name"
                    scope="col"
                    class="px-3 py-1 text-center font-bold whitespace-nowrap text-ink-soft"
                  >
                    {{ name }}
                  </th>
                  <th scope="col" class="px-3 py-1 text-center font-bold text-ink-soft">
                    {{ t('visualize.total') }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(name, row) in cross.rows" :key="name">
                  <th
                    scope="row"
                    class="py-1 pr-3 text-left font-bold whitespace-nowrap text-ink-soft"
                  >
                    {{ name }}
                  </th>
                  <td
                    v-for="(other, cell) in cross.columns"
                    :key="other"
                    class="px-3 py-1 text-center tabular-nums"
                    :style="crossStyle(cross.counts[row]?.[cell] ?? 0)"
                  >
                    {{ cross.counts[row]?.[cell] ?? 0 }}
                  </td>
                  <td class="px-3 py-1 text-center font-bold tabular-nums">
                    {{ crossRowTotals[row] }}
                  </td>
                </tr>
                <tr>
                  <th scope="row" class="py-1 pr-3 text-left font-bold text-ink-soft">
                    {{ t('visualize.total') }}
                  </th>
                  <td
                    v-for="(total, cell) in crossColumnTotals"
                    :key="cell"
                    class="px-3 py-1 text-center font-bold tabular-nums"
                  >
                    {{ total }}
                  </td>
                  <td class="px-3 py-1 text-center font-bold tabular-nums">{{ crossTotal }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </AppCard>
    </template>
  </div>
</template>
