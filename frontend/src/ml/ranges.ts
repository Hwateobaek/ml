/**
 * 사람이 정한 범위 — 이상치 처리의 `range` (open-decisions.md "이상치는 훈련 데이터의
 * IQR로 클리핑한다"의 "사람이 정한 범위로는 행을 뺀다").
 *
 * **경계를 데이터에서 구하지 않는다.** 그래서 분할 전(`usableRows`)에 써도 누수가 없고,
 * 결측치 `drop`과 같은 자리에서 행을 뺀다. 클리핑(`ml/outliers.ts`)과 갈리는 것이 정확히
 * 이 점이다.
 *
 * 검사는 `tests/ranges.spec.ts`가 붙는다.
 */

/** 열 하나의 범위. **한쪽만 있어도 된다.** 경계값은 범위 안이다. */
export type ColumnRange = {
  readonly min?: number | undefined
  readonly max?: number | undefined
}

/** 원본 열 이름 → 범위. 키는 `features`·`target`과 같은 원본 이름이다. */
export type ColumnRanges = Readonly<Record<string, ColumnRange>>

/**
 * 지금 행을 빼는 범위. **`range`를 고른 동안만 있다.**
 *
 * 다른 방식으로 바꿔도 적어 둔 숫자는 파일에 남는다 — 다시 고르면 되살아난다. 그 숫자가
 * 행을 빼지 않게 막는 것이 여기다. `usableRows`를 부르는 자리는 전부 이것을 지난다.
 */
export function activeRanges(preprocessing: {
  readonly outliers?: string | undefined
  readonly ranges?: ColumnRanges | undefined
}): ColumnRanges | undefined {
  return preprocessing.outliers === 'range' ? preprocessing.ranges : undefined
}

/** 적힌 것이 하나라도 있는가. 둘 다 비었으면 없는 범위다. */
export function hasBound(range: ColumnRange | undefined): boolean {
  return range !== undefined && (range.min !== undefined || range.max !== undefined)
}

/** 범위 안인가. 경계값은 안이다. */
export function inRange(value: number, range: ColumnRange): boolean {
  if (range.min !== undefined && value < range.min) return false
  if (range.max !== undefined && value > range.max) return false
  return true
}

/**
 * 이 범위를 쓸 수 있는가. 쓸 수 있으면 `null`이다.
 *
 * **boolean이 아니라 이유다** (`CLAUDE.md` §2). 뒤집힌 범위는 모든 행을 뺀다 — 학습이
 * `SPLIT_TOO_FEW_ROWS`로 서기 전에 입력칸에서 말해야 학생이 무엇을 고칠지 안다.
 */
export function rangeProblem(range: ColumnRange): 'reversed' | null {
  if (range.min !== undefined && range.max !== undefined && range.min > range.max) {
    return 'reversed'
  }
  return null
}

/**
 * 범위 하나를 적어 넣은 **새 대응표**. 원본은 안 건드린다.
 *
 * **둘 다 비면 항목을 지운다.** 남겨 두면 "범위 없음"이 두 모양으로 파일에 남고, 견줄 때
 * 아무것도 안 바꾼 학생에게 변경이 뜬다.
 */
export function withColumnRange(
  ranges: ColumnRanges | undefined,
  column: string,
  range: ColumnRange,
): Record<string, ColumnRange> {
  const rest: Record<string, ColumnRange> = { ...(ranges ?? {}) }
  delete rest[column]
  if (!hasBound(range)) return rest
  return {
    ...rest,
    [column]: {
      ...(range.min === undefined ? {} : { min: range.min }),
      ...(range.max === undefined ? {} : { max: range.max }),
    },
  }
}

/**
 * 지금 쓰는 열에 걸린 범위만, 이름 순서로. **행을 빼는 것과 견주는 것이 같은 목록을 본다.**
 *
 * **특성에서 뺀 열의 범위는 아무 행도 안 뺀다.** 안 그러면 화면에 없는 열이 행을 지운다.
 */
export function appliedRanges(
  ranges: ColumnRanges | undefined,
  features: readonly string[],
  target: string | undefined,
): (readonly [string, ColumnRange])[] {
  if (ranges === undefined) return []
  const used = new Set(target === undefined ? features : [...features, target])
  return Object.entries(ranges)
    .filter(([column, range]) => used.has(column) && hasBound(range))
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
}

/**
 * 입력칸의 글자를 경계로 읽는다. **비었으면 `undefined`(그쪽은 제한 없음), 숫자가 아니면
 * `null`이다.**
 *
 * `null`과 `undefined`를 가르는 이유는 둘이 학생에게 다른 말이기 때문이다 — 비운 것은
 * 뜻이 있는 선택이고, `abc`는 고칠 입력이다. 천 단위 쉼표는 받는다 (`1,000`).
 */
export function parseBound(text: string): number | undefined | null {
  const trimmed = text.trim().replace(/,/g, '')
  if (trimmed === '') return undefined
  const value = Number(trimmed)
  return Number.isFinite(value) ? value : null
}

/**
 * 변경 이력에 적을 한 줄. `Body Mass (g) [3000, 5000]` 꼴이다.
 *
 * **구간 기호를 쓴다.** 언어가 없는 표기라 이 계층이 만들어도 된다 — 숫자도 `String`으로만
 * 편다 (`ml/changes.ts`의 `literal`과 같다). 한쪽이 비면 무한대 기호로 적는다.
 */
export function describeRange([column, range]: readonly [string, ColumnRange]): string {
  const low = range.min === undefined ? '−∞' : String(range.min)
  const high = range.max === undefined ? '∞' : String(range.max)
  return `${column} [${low}, ${high}]`
}
