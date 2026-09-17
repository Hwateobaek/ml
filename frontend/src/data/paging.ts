/**
 * 쪽 나눔. **행을 자르는 계산만 있고 화면은 모른다.**
 *
 * 화면 밖에 있는 이유는 경계값이 눈으로 안 보이기 때문이다 — 마지막 쪽에 한 행만 남는
 * 경우, 행이 하나도 없는 경우, 쪽 번호가 범위를 벗어난 경우. 셋 다 그려 놓으면 그럴듯해
 * 보이고, 틀리면 학생이 자기 데이터의 마지막 몇 행을 영영 못 본다.
 *
 * 검사는 `tests/paging.spec.ts`가 붙는다.
 */

/** 이 행 수를 이 크기로 나누면 몇 쪽인가. **행이 없어도 한 쪽이다** — 빈 표도 자리는 있다. */
export function pageCount(total: number, size: number): number {
  if (size <= 0) return 1
  return Math.max(1, Math.ceil(total / size))
}

/** 쪽 번호를 있는 범위 안으로 밀어 넣는다. 쪽은 1부터 센다. */
export function clampPage(page: number, total: number, size: number): number {
  const last = pageCount(total, size)
  if (!Number.isFinite(page)) return 1
  return Math.min(Math.max(Math.trunc(page), 1), last)
}

/** 이 쪽이 덮는 구간 `[start, end)`. `start`는 0부터 센다. */
export function pageRange(
  page: number,
  total: number,
  size: number,
): { start: number; end: number } {
  const current = clampPage(page, total, size)
  const start = (current - 1) * size
  return { start, end: Math.min(start + size, total) }
}

/** 이 쪽에 그릴 것들. 원본 배열은 안 건드린다. */
export function pageItems<Item>(
  items: readonly Item[],
  page: number,
  size: number,
): readonly Item[] {
  const { start, end } = pageRange(page, items.length, size)
  return items.slice(start, end)
}
