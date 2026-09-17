/**
 * `data/missing.ts` — [결측치 표시]가 칠할 행을 고르는 규칙.
 *
 * **요약표의 "결측치 수"와 같은 뜻인지를 지킨다.** 둘이 갈리면 색칠된 칸과 숫자가 어긋난다.
 */

import { describe, expect, it } from 'vitest'

import { summarizeColumns } from '@/data/columns'
import { isBlankCell, missingRowCount, rowHasMissing } from '@/data/missing'
import type { Dataset } from '@/ml/preprocess'

describe('isBlankCell', () => {
  it('빈 칸과 공백뿐인 칸이 결측이다', () => {
    expect(isBlankCell('')).toBe(true)
    expect(isBlankCell('   ')).toBe(true)
    expect(isBlankCell(undefined)).toBe(true)
  })

  it('N/A 같은 글자와 0은 값이다', () => {
    expect(isBlankCell('N/A')).toBe(false)
    expect(isBlankCell('0')).toBe(false)
  })
})

describe('rowHasMissing', () => {
  it('칸 하나라도 비면 참이다', () => {
    expect(rowHasMissing(['1', '', 'a'], 3)).toBe(true)
    expect(rowHasMissing(['1', '2', 'a'], 3)).toBe(false)
  })

  it('머리글보다 짧은 행은 모자란 칸이 결측이다', () => {
    expect(rowHasMissing(['1', '2'], 3)).toBe(true)
  })
})

describe('missingRowCount', () => {
  const table: Dataset = {
    columns: ['a', 'b'],
    rows: [
      ['1', '2'],
      ['', '2'],
      ['1', ' '],
      ['', ''],
    ],
  }

  it('결측치가 있는 행을 센다 — 한 행의 빈 칸이 여럿이어도 한 행이다', () => {
    expect(missingRowCount(table)).toBe(3)
  })

  it('요약표의 결측치 수와 같은 칸을 빈 칸으로 본다', () => {
    const blankCells = table.rows
      .flatMap((row) => table.columns.map((_, index) => row[index]))
      .filter(isBlankCell).length
    const summarized = summarizeColumns(table).reduce((sum, column) => sum + column.missing, 0)
    expect(blankCells).toBe(summarized)
  })
})
