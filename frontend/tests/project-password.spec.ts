// @vitest-environment jsdom
/**
 * 프로젝트 잠금이 **열기 길목에** 걸려 있는가 (stores/project.ts의 open·unlock,
 * open-decisions.md "프로젝트에 잠금을 걸 수 있다 — 파일에는 안 들어간다").
 *
 * 비밀번호 자체의 셈은 `lock.spec.ts`가 잰다. 여기서 재는 것은 배선이다 —
 * 잠긴 것을 **읽기 전에** 막는가(막지 않으면 주소를 치는 것으로 내용이 열린다),
 * 통과하면 열리는가, 그리고 **통과가 그 프로젝트에만** 미치는가.
 */

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { makeLock } from '../src/project/lock'
import { useProjectStore } from '../src/stores/project'
import { useToastStore } from '../src/stores/toasts'

vi.mock('../src/project/tab-lock', () => ({
  acquireTabLock: vi.fn(async () => true),
  releaseTabLock: vi.fn(),
}))

const loadProject = vi.fn()
const readLock = vi.fn<(id: string) => Promise<unknown>>()

vi.mock('../src/project/storage', () => ({
  loadProject: (id: string) => loadProject(id) as Promise<unknown>,
  markExported: vi.fn(),
  readExportedAt: vi.fn(async () => null),
  readLock: (id: string) => readLock(id),
  requestPersistence: vi.fn(async () => false),
  saveProject: vi.fn(),
}))

/** 읽기가 성공했을 때 돌려줄 최소한의 프로젝트. */
const opened = {
  document: { manifest: { projectId: 'p-1', updatedAt: '2026-11-05T00:00:00.000Z' } },
}

beforeEach(() => {
  setActivePinia(createPinia())
  vi.clearAllMocks()
  loadProject.mockResolvedValue(opened)
  readLock.mockResolvedValue(null)
})

describe('잠긴 프로젝트를 연다', () => {
  it('안 잠겼으면 묻지 않고 연다 — 지금까지와 같다', async () => {
    const project = useProjectStore()
    expect(await project.open('p-1')).toBe(true)
    expect(useToastStore().items).toHaveLength(0)
  })

  /**
   * **읽기 전에 막아야 한다.** 읽고 나서 막으면 내용이 이미 메모리에 올라와 있고,
   * 자동 저장이 그것을 물고 들어온다 (탭 잠금이 같은 이유로 읽기 전에 선다).
   */
  it('잠겼으면 읽기 전에 돌아서고 그 사실을 말한다', async () => {
    readLock.mockResolvedValue(makeLock('1234'))
    const project = useProjectStore()

    expect(await project.open('p-1')).toBe(false)
    expect(loadProject).not.toHaveBeenCalled()
    expect(project.file).toBeNull()

    const toasts = useToastStore().items
    expect(toasts).toHaveLength(1)
    expect(toasts[0]).toMatchObject({ tone: 'danger', key: 'client.PROJECT_LOCKED' })
  })

  it('비밀번호가 맞으면 그 뒤로 열린다', async () => {
    const lock = makeLock('1234')
    readLock.mockResolvedValue(lock)
    const project = useProjectStore()

    expect(lock && project.unlock('p-1', '1234', lock)).toBe(true)
    expect(await project.open('p-1')).toBe(true)
    expect(project.file).not.toBeNull()
  })

  it('비밀번호가 틀리면 통과하지 않고, 열기도 여전히 막힌다', async () => {
    const lock = makeLock('1234')
    readLock.mockResolvedValue(lock)
    const project = useProjectStore()

    expect(lock && project.unlock('p-1', '0000', lock)).toBe(false)
    expect(await project.open('p-1')).toBe(false)
    expect(loadProject).not.toHaveBeenCalled()
  })

  /**
   * **한 번 통과했다고 다른 프로젝트까지 열리면 안 된다.** 통과는 id마다 따로 센다.
   */
  it('통과는 그 프로젝트에만 미친다', async () => {
    const lock = makeLock('1234')
    readLock.mockResolvedValue(lock)
    const project = useProjectStore()

    expect(lock && project.unlock('p-1', '1234', lock)).toBe(true)
    expect(await project.open('p-2')).toBe(false)
  })
})
