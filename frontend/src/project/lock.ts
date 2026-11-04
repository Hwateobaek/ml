/**
 * 프로젝트 잠금 (open-decisions.md "프로젝트에 잠금을 걸 수 있다 — 파일에는 안 들어간다").
 *
 * **순수 함수다.** IndexedDB도 화면도 모른다 — 거는 자리는 `project/storage.ts`이고
 * 묻는 자리는 `stores/project.ts`다.
 *
 * **이것은 암호화가 아니라 화면 앞의 문이다.** 막는 것은 공용 PC에서 옆자리 학생이
 * 남의 프로젝트를 여는 일이고, devtools로 IndexedDB를 읽는 것은 못 막는다. 내용을
 * 정말 숨기려면 암호화해야 하는데 그러면 목록이 이름도 크기도 못 읽어 "못 읽는 것도
 * 목록에는 남긴다"(architecture.md §8.10.2)가 깨진다.
 *
 * **`.mlpx`에는 안 들어간다.** 파일로 받은 프로젝트는 비밀번호 없이 열린다 —
 * 교사는 파일 하나만 열면 다 봐야 한다 (CLAUDE.md §1.3).
 */

import { hashText } from '@/hash'

/**
 * 레코드에 저장되는 잠금. **비밀번호는 여기 없다.**
 *
 * 힌트는 선택이다. 비밀번호를 잊으면 그 프로젝트는 영영 안 열리므로 학생이 스스로
 * 단서를 남길 자리를 준다. **비밀번호 자체를 저장하지 않으므로 힌트를 두어도
 * 안전성은 그대로다** — 힌트를 읽어도 비밀번호가 나오지는 않는다.
 */
export interface ProjectLock {
  /** 소금. 없으면 두 프로젝트의 비밀번호가 같은지 저장된 값만 보고도 알 수 있다. */
  readonly salt: string
  /** `salt + '|' + password`의 SHA-256. */
  readonly hash: string
  /** 학생이 적은 단서. 안 적었으면 없다. */
  readonly hint?: string
}

/**
 * 소금 한 조각.
 *
 * **`crypto.randomUUID`를 쓰지 않는다** — 보안 컨텍스트(https, localhost)에서만
 * 존재하는데 자가호스팅은 대개 http로 접속한다 (`project/create.ts`의 `newProjectId`와
 * 같은 이유, 같은 배포 경로).
 */
function newSalt(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

/**
 * 비밀번호를 해시로 바꾼다. **같은 비밀번호라도 소금이 다르면 값이 다르다.**
 *
 * `crypto.subtle`이 아니라 `hash.ts`를 지난다 — 그쪽은 보안 컨텍스트를 요구하고,
 * 이 저장소는 이미 그 길을 버렸다.
 */
function digest(salt: string, password: string): string {
  return hashText(`${salt}|${password}`)
}

/**
 * 빈 비밀번호는 잠금이 아니다.
 *
 * **입구에서 거른다.** 공백만 적은 것을 잠금으로 받아들이면 학생은 잠갔다고 믿는데
 * 아무나 빈 칸으로 열 수 있다.
 */
export function isUsablePassword(password: string): boolean {
  return password.trim() !== ''
}

/**
 * 새 잠금을 만든다. 쓸 수 없는 비밀번호면 `null`이다 — 부르는 쪽이 "안 걸었다"로 읽는다.
 *
 * **힌트의 공백은 떼고, 비어 있으면 키 자체를 안 넣는다.** `undefined`를 넣어 두면
 * "안 적음"과 "빈 문자열"이 레코드 안에서 섞인다 (`storage.ts`의 `exportedAt`과 같은 규칙).
 */
export function makeLock(password: string, hint = ''): ProjectLock | null {
  if (!isUsablePassword(password)) return null
  const salt = newSalt()
  const trimmed = hint.trim()
  return {
    salt,
    hash: digest(salt, password),
    ...(trimmed === '' ? {} : { hint: trimmed }),
  }
}

/**
 * 비밀번호가 맞는가.
 *
 * **비교는 길이부터 본다.** 해시는 비밀이 아니라 큰 뜻은 없지만, 길이가 다른 것을
 * 글자마다 견줄 이유도 없다.
 */
export function checkLock(lock: ProjectLock, password: string): boolean {
  const expected = digest(lock.salt, password)
  return expected.length === lock.hash.length && expected === lock.hash
}

/**
 * 레코드에서 읽은 값이 잠금의 모양인가.
 *
 * **zod를 안 쓴다.** 이 판정은 목록을 그릴 때마다 프로젝트 수만큼 돈다 —
 * `listProjects`가 같은 이유로 스키마를 안 돌린다. 볼 것은 두 글자열뿐이다.
 *
 * **모양이 아니면 잠기지 않은 것으로 친다.** 손상된 잠금으로 문을 닫으면 학생은
 * 맞는 비밀번호를 넣고도 못 들어가고, 그 프로젝트를 되살릴 길이 없다.
 */
export function isLock(value: unknown): value is ProjectLock {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as { salt?: unknown; hash?: unknown; hint?: unknown }
  if (typeof candidate.salt !== 'string' || candidate.salt === '') return false
  if (typeof candidate.hash !== 'string' || candidate.hash === '') return false
  return candidate.hint === undefined || typeof candidate.hint === 'string'
}
