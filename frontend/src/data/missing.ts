/**
 * 표에서 **결측치가 있는 행**을 찾는다 — 데이터 미리보기와 전체 데이터 보기의 [결측치 표시]가 쓴다.
 *
 * **결측의 뜻은 열 요약과 같다** (`data/columns.ts`의 `summarizeColumns`): 빈 칸이거나
 * 공백뿐인 칸이다. `N/A` 같은 글자는 값이다 — 여기서 다르게 세면 요약표의 "결측치 수"와
 * 색칠된 칸의 수가 어긋나고, 학생은 어느 쪽을 믿어야 할지 모른다.
 *
 * **행이 머리글보다 짧으면 모자란 칸도 결측이다.** 요약표가 `row[index] ?? ''`로 세는
 * 것과 같다.
 *
 * 검사는 `tests/missing.spec.ts`가 붙는다.
 */

import type { Dataset } from '@/ml/preprocess'

/** 빈 칸인가. */
export function isBlankCell(cell: string | undefined): boolean {
  return cell === undefined || cell.trim() === ''
}

/** 이 행에 결측치가 하나라도 있는가. `columnCount`는 머리글의 열 수다. */
export function rowHasMissing(row: readonly string[], columnCount: number): boolean {
  for (let index = 0; index < columnCount; index += 1) {
    if (isBlankCell(row[index])) return true
  }
  return false
}

/** 결측치가 있는 행의 수. */
export function missingRowCount(dataset: Dataset): number {
  const columnCount = dataset.columns.length
  let count = 0
  for (const row of dataset.rows) if (rowHasMissing(row, columnCount)) count += 1
  return count
}
