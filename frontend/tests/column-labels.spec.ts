/**
 * `data/column-labels.ts` — 학생이 고쳐 부르는 열 이름.
 *
 * **여기서 지키는 것은 "원본이 안 바뀐다"이다.** 대응표가 원본 이름을 키로 들고 있는
 * 한 타깃·특성·전처리가 안 풀리는데, 그 성질은 화면에서는 안 보이고 **이름을 고친
 * 다음에야** 드러난다.
 */

import { describe, expect, it } from 'vitest'

import {
  columnLabel,
  columnLabelProblem,
  columnLabelsOf,
  withColumnLabel,
  type ColumnLabels,
} from '@/data/column-labels'

const columns = ['Culmen Length (mm)', 'Culmen Depth (mm)', 'Species']
const labels: ColumnLabels = { 'Culmen Length (mm)': '부리 길이(mm)' }

describe('columnLabel', () => {
  it('고친 열은 부르는 이름으로 나온다', () => {
    expect(columnLabel('Culmen Length (mm)', labels)).toBe('부리 길이(mm)')
  })

  it('안 고친 열은 원본 이름 그대로다', () => {
    expect(columnLabel('Species', labels)).toBe('Species')
  })

  it('대응표가 없어도 원본 이름을 부른다 - 예전 형식으로 저장된 파일이 그 자리다', () => {
    expect(columnLabel('Species', undefined)).toBe('Species')
  })

  it('빈 문자열이 적혀 있어도 원본 이름을 부른다', () => {
    expect(columnLabel('Species', { Species: '' })).toBe('Species')
  })
})

describe('columnLabelsOf', () => {
  it('순서를 바꾸지 않는다', () => {
    expect(columnLabelsOf(columns, labels)).toEqual([
      '부리 길이(mm)',
      'Culmen Depth (mm)',
      'Species',
    ])
  })
})

describe('columnLabelProblem', () => {
  it('다른 열이 이미 그렇게 불리면 막는다', () => {
    expect(columnLabelProblem(columns, labels, 'Species', '부리 길이(mm)')).toBe('duplicate')
  })

  it('다른 열의 원본 이름과 겹쳐도 막는다', () => {
    expect(columnLabelProblem(columns, labels, 'Species', 'Culmen Depth (mm)')).toBe('duplicate')
  })

  it('자기 이름을 그대로 두는 것은 겹침이 아니다', () => {
    expect(columnLabelProblem(columns, labels, 'Species', 'Species')).toBeNull()
  })

  it('비우는 것은 오류가 아니다 - 원본으로 돌아간다는 뜻이다', () => {
    expect(columnLabelProblem(columns, labels, 'Species', '   ')).toBeNull()
  })
})

describe('withColumnLabel', () => {
  it('넘겨받은 대응표를 안 건드리고 새것을 준다', () => {
    const before: ColumnLabels = { ...labels }
    withColumnLabel(before, 'Species', '품종')
    expect(before).toEqual(labels)
  })

  it('앞뒤 공백을 떼고 적는다', () => {
    expect(withColumnLabel(labels, 'Species', '  품종  ')['Species']).toBe('품종')
  })

  it('비우면 항목이 사라진다', () => {
    const next = withColumnLabel(labels, 'Culmen Length (mm)', '')
    expect('Culmen Length (mm)' in next).toBe(false)
  })

  it('원본 이름과 같아지면 항목을 안 남긴다', () => {
    const next = withColumnLabel(labels, 'Culmen Length (mm)', 'Culmen Length (mm)')
    expect('Culmen Length (mm)' in next).toBe(false)
  })

  it('다른 열의 이름은 그대로 둔다', () => {
    const next = withColumnLabel(labels, 'Species', '품종')
    expect(next['Culmen Length (mm)']).toBe('부리 길이(mm)')
  })
})
