<script setup lang="ts">
/**
 * 합성곱이 **무엇을 보는가** (open-decisions.md "필터와 특성맵을 보여준다").
 *
 * 두 가지를 그린다.
 *
 * 1. **필터** — 첫 층이 찾는 무늬. 가중치를 그대로 색으로 옮긴 것이고, 학생이 눈으로
 *    볼 수 있는 **유일한 층**이다(둘째 층부터는 들어오는 칸이 색이 아니다).
 * 2. **특성맵** — 사진 한 장이 층을 지나며 어떻게 변하는가. 밝은 곳이 **그 필터가
 *    반응한 자리**다.
 *
 * **계산은 `ml/models/cnn.ts`에 있다.** 예측이 지나는 그 함수를 그대로 지나므로
 * 화면이 **모델이 보지 않은 그림**을 보여줄 수 없다. 여기 있는 것은 그리기뿐이다 (§8.3).
 *
 * **사진은 훈련에 쓴 것 중 첫 장이다.** 고르게 하지 않는다 — 손잡이가 하나 늘고,
 * 학생이 답해야 하는 질문("어느 사진으로 볼까")이 수업 내용이 아니다.
 */

import { computed, ref, watchEffect } from 'vue'
import { useI18n } from 'vue-i18n'

import { CNN_FEATURE_MAP_COUNT } from '@/limits'
import { cnnFeatureMaps, cnnFirstFilters, cnnModelOf, type CnnFeatureMaps } from '@/ml/cnn-view'
import type { PanelInput } from '@/ml/metric-panels'
import { canvasPixelDecoder } from '@/ml/pixel-draw'
import { readImages } from '@/project/images'
import { IMAGE_UNLABELED } from '@/project/format'
import { useProjectStore } from '@/stores/project'

const props = defineProps<{ input: PanelInput }>()

const { t } = useI18n()
const project = useProjectStore()

const model = computed(() => cnnModelOf(props.input.run.model?.format, props.input.modelBytes))

/** 첫 층의 필터들. **몇 장을 그릴지는 `limits.ts`가 정한다.** */
const filters = computed(() => {
  const parsed = model.value
  if (!parsed) return null
  const { size, filters: tiles } = cnnFirstFilters(parsed)
  return { size, tiles: tiles.slice(0, CNN_FEATURE_MAP_COUNT), total: tiles.length }
})

/**
 * 특성맵을 뽑을 사진. **라벨 붙은 첫 장이다** — 라벨 없는 사진은 학습에 안 들어갔으므로
 * 그것으로 그리면 "학습한 모델이 본 것"이 아니다.
 */
const photo = computed(() => {
  const file = project.file
  if (!file) return null
  return readImages(file).find((entry) => entry.category !== IMAGE_UNLABELED) ?? null
})

const maps = ref<CnnFeatureMaps[] | null>(null)
const photoUrl = ref<string | null>(null)

/**
 * 사진을 화소로 펴서 층마다의 그림을 얻는다.
 *
 * **학습 때와 같은 함수로 편다** (`ml/pixel-draw.ts`) — 갈리면 화면이 모델이 보지 않은
 * 사진을 보여준다.
 */
watchEffect(async (onCleanup) => {
  const parsed = model.value
  const entry = photo.value
  maps.value = null
  if (!parsed || !entry) return

  const url = URL.createObjectURL(new Blob([entry.bytes as Uint8Array<ArrayBuffer>]))
  photoUrl.value = url
  onCleanup(() => {
    URL.revokeObjectURL(url)
  })

  let cancelled = false
  onCleanup(() => {
    cancelled = true
  })
  try {
    const [row] = await canvasPixelDecoder([entry.bytes])
    if (cancelled || !row) return
    maps.value = cnnFeatureMaps(parsed, row)
  } catch {
    // 못 그리면 안 그린다 (§9.2). 사유를 말하는 자리는 다른 곳이다.
    maps.value = null
  }
})

/** 필터 타일 하나를 캔버스에 그린다. **가중치는 음수도 되므로 범위를 맞춰 옮긴다.** */
function drawFilter(canvas: HTMLCanvasElement | null, tile: Float32Array, size: number): void {
  if (!canvas) return
  const context = canvas.getContext('2d')
  if (!context) return
  let low = Infinity
  let high = -Infinity
  for (const value of tile) {
    if (value < low) low = value
    if (value > high) high = value
  }
  // 전부 같은 값이면 나눌 것이 없다. 회색으로 둔다.
  const span = high - low || 1
  const image = context.createImageData(size, size)
  for (let at = 0; at < size * size; at += 1) {
    for (let c = 0; c < 3; c += 1) {
      image.data[at * 4 + c] = Math.round(((tile[at * 3 + c]! - low) / span) * 255)
    }
    image.data[at * 4 + 3] = 255
  }
  canvas.width = size
  canvas.height = size
  context.putImageData(image, 0, 0)
}

/**
 * 특성맵 한 장을 그린다. **회색조다** — 한 장이 필터 하나의 반응이라 색이 아니고,
 * 색을 입히면 없는 뜻을 보탠다.
 */
function drawMap(canvas: HTMLCanvasElement | null, map: CnnFeatureMaps, channel: number): void {
  if (!canvas) return
  const context = canvas.getContext('2d')
  if (!context) return
  const { side, channels, values } = map
  let high = 0
  for (let at = 0; at < side * side; at += 1) {
    const value = values[at * channels + channel]!
    if (value > high) high = value
  }
  // **ReLU를 지나 왔으므로 아래쪽 끝은 0이다.** 전부 0이면 까맣게 둔다 — 그 필터가
  // 이 사진에서 아무것도 못 봤다는 뜻이고, 그것도 볼거리다.
  const span = high || 1
  const image = context.createImageData(side, side)
  for (let at = 0; at < side * side; at += 1) {
    const shade = Math.round((values[at * channels + channel]! / span) * 255)
    image.data[at * 4] = shade
    image.data[at * 4 + 1] = shade
    image.data[at * 4 + 2] = shade
    image.data[at * 4 + 3] = 255
  }
  canvas.width = side
  canvas.height = side
  context.putImageData(image, 0, 0)
}
</script>

<template>
  <section v-if="filters && filters.tiles.length > 0" class="flex flex-col gap-5">
    <div class="flex flex-col gap-1.5">
      <h4 class="text-base font-bold">{{ t('results.cnn.filtersTitle') }}</h4>
      <p class="text-base text-ink-soft">{{ t('results.cnn.filtersLead') }}</p>
      <p v-if="filters.total > filters.tiles.length" class="text-base text-ink-faint">
        {{ t('results.cnn.shown', { shown: filters.tiles.length, total: filters.total }) }}
      </p>
    </div>

    <!--
      **타일이 작아서 늘려 그린다.** 3×3 캔버스를 CSS로 키우면 흐려지므로
      `image-rendering: pixelated`로 화소를 그대로 보인다 (`styles/utilities.css`).
    -->
    <ul class="flex flex-wrap gap-2">
      <li v-for="(tile, index) in filters.tiles" :key="index">
        <canvas
          :ref="(el) => drawFilter(el as HTMLCanvasElement | null, tile, filters!.size)"
          class="size-12 rounded border border-line-strong bg-surface pixelated"
          :aria-label="t('results.cnn.filterLabel', { index: index + 1 })"
        />
      </li>
    </ul>

    <template v-if="maps && maps.length > 0">
      <div class="flex flex-col gap-1.5">
        <h4 class="text-base font-bold">{{ t('results.cnn.mapsTitle') }}</h4>
        <p class="text-base text-ink-soft">{{ t('results.cnn.mapsLead') }}</p>
      </div>

      <div class="flex flex-col gap-5">
        <div class="flex flex-col gap-1.5">
          <p class="text-base font-bold text-ink-soft">{{ t('results.cnn.original') }}</p>
          <img
            v-if="photoUrl"
            :src="photoUrl"
            :alt="t('results.cnn.original')"
            class="size-24 rounded border border-line-strong object-cover"
          />
        </div>

        <div v-for="map in maps" :key="map.block" class="flex flex-col gap-1.5">
          <p class="text-base font-bold text-ink-soft">
            {{ t('results.cnn.blockLabel', { block: map.block, side: map.side }) }}
          </p>
          <ul class="flex flex-wrap gap-2">
            <li v-for="channel in Math.min(map.channels, CNN_FEATURE_MAP_COUNT)" :key="channel">
              <canvas
                :ref="(el) => drawMap(el as HTMLCanvasElement | null, map, channel - 1)"
                class="size-12 rounded border border-line-strong bg-surface pixelated"
                :aria-label="t('results.cnn.mapLabel', { block: map.block, index: channel })"
              />
            </li>
          </ul>
        </div>
      </div>
    </template>
  </section>
</template>
