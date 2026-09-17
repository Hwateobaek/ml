/**
 * 화면이 열 이름을 **부르는 이름으로** 찍는 통로.
 *
 * **화면마다 스토어를 뒤지지 않게 하는 자리다.** 열 이름은 전처리·학습·결과·예측·
 * 시각화가 전부 찍는 것이라, 각자 `tabularDataOf(...)?.columnLabels`를 꺼내 쓰면
 * 그 표기가 스무 군데로 흩어진다. 흩어지면 **한 군데는 반드시 빠지고**, 빠진 자리는
 * 조용히 원본 이름을 찍어서 눈으로만 보면 고장으로 안 보인다.
 *
 * 계산은 `data/column-labels.ts`에 있고 (`tests/column-labels.spec.ts`가 지킨다)
 * 여기는 그것을 지금 열린 프로젝트에 이어 준다.
 */

import { computed } from 'vue'

import { columnLabel, columnLabelsOf, type ColumnLabels } from '@/data/column-labels'
import { tabularDataOf } from '@/project/schema'
import { useProjectStore } from '@/stores/project'

export function useColumnLabels() {
  const project = useProjectStore()

  /** 이미지 프로젝트에는 열이 없다 — 그때 `tabularDataOf`가 `null`이고 대응표는 비었다. */
  const labels = computed<ColumnLabels>(
    () => tabularDataOf(project.file?.document)?.columnLabels ?? {},
  )

  return {
    labels,
    /** 열 하나의 부르는 이름. 안 고쳤으면 원본 그대로다. */
    label: (name: string): string => columnLabel(name, labels.value),
    /** 여러 열을 한꺼번에. 순서는 그대로다. */
    labelsFor: (names: readonly string[]): string[] => columnLabelsOf(names, labels.value),
  }
}
