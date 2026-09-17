/**
 * 학생이 고쳐 부르는 열 이름.
 *
 * **원본은 절대 안 바뀐다** (`CLAUDE.md` §1.3, `project/format.ts`의 `Dataset.bytes`가
 * "업로드된 원본 그대로. 절대 가공하지 않는다"). 여기 있는 것은 **원본 이름 → 부르는
 * 이름**의 대응표 하나이고, 계산과 저장은 전부 원본 이름을 키로 그대로 쓴다.
 *
 * **그래서 이름을 고쳐도 고른 것이 안 풀린다.** 타깃·특성·전처리·실험 스냅샷이 전부
 * 원본 이름을 들고 있으므로, 대응표는 그것들을 하나도 안 건드린다. 갈리는 것은 화면에
 * 찍히는 글자뿐이다 — 대응표를 통째로 지워도 프로젝트는 그대로 학습된다.
 *
 * 검사는 `tests/column-labels.spec.ts`가 붙는다.
 */

/** 원본 이름 → 부르는 이름. 안 고친 열은 여기 없다. */
export type ColumnLabels = Readonly<Record<string, string>>

/**
 * 이 열을 화면에서 뭐라고 부르는가. **안 고쳤으면 원본 이름 그대로다.**
 *
 * 빈 대응표와 없는 대응표를 갈라 다루지 않는다 — 아직 아무것도 안 고친 프로젝트와
 * 예전 형식으로 저장된 파일이 같은 자리이고, 둘 다 원본 이름을 부른다.
 */
export function columnLabel(name: string, labels: ColumnLabels | undefined): string {
  const label = labels?.[name]
  return label === undefined || label === '' ? name : label
}

/** 열 이름들을 한꺼번에 옮긴다. 순서는 그대로다. */
export function columnLabelsOf(
  names: readonly string[],
  labels: ColumnLabels | undefined,
): string[] {
  return names.map((name) => columnLabel(name, labels))
}

/**
 * 이 이름을 쓸 수 있는가. 쓸 수 있으면 `null`이다.
 *
 * **boolean이 아니라 이유다** (`CLAUDE.md` §2의 gate 규칙). 화면이 "안 된다"만 알면
 * 학생에게 할 말이 하나뿐이고, 비어서 안 되는 것과 겹쳐서 안 되는 것은 고치는 방법이
 * 서로 다르다.
 *
 * **비어 있는 것은 오류가 아니다** — 지우면 원본 이름으로 돌아간다는 뜻이라 `null`이다.
 * 막아야 하는 것은 **다른 열이 이미 그렇게 불리는 경우** 하나다. 같은 이름 둘이 화면에
 * 나란히 서면 어느 것을 고르는지 학생이 알 수 없다.
 */
export function columnLabelProblem(
  columns: readonly string[],
  labels: ColumnLabels | undefined,
  original: string,
  next: string,
): 'duplicate' | null {
  const trimmed = next.trim()
  if (trimmed === '') return null
  const taken = columns.some((name) => name !== original && columnLabel(name, labels) === trimmed)
  return taken ? 'duplicate' : null
}

/**
 * 대응표에 하나를 적어 넣은 **새 대응표**를 준다. 원본은 안 건드린다.
 *
 * **원본 이름과 같아지면 항목을 지운다.** 남겨 두면 대응표가 "안 고친 것"과 "고쳐서
 * 원래대로 돌아온 것"을 갈라 들게 되는데, 화면에 나오는 결과가 같아서 그 구분은 아무
 * 데도 쓰이지 않고 파일만 커진다.
 */
export function withColumnLabel(
  labels: ColumnLabels | undefined,
  original: string,
  next: string,
): ColumnLabels {
  const trimmed = next.trim()
  const rest = { ...(labels ?? {}) }
  delete rest[original]
  if (trimmed === '' || trimmed === original) return rest
  return { ...rest, [original]: trimmed }
}
