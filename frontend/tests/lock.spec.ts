import { describe, expect, it } from 'vitest'

import { checkLock, isLock, isUsablePassword, makeLock } from '../src/project/lock'

describe('잠금을 만든다', () => {
  it('비밀번호가 비어 있으면 잠그지 않는다', () => {
    expect(makeLock('')).toBeNull()
    expect(makeLock('   ')).toBeNull()
  })

  it('빈 비밀번호를 입구에서 거른다 - 잠갔다고 믿는데 빈 칸으로 열리면 안 된다', () => {
    expect(isUsablePassword('')).toBe(false)
    expect(isUsablePassword(' \t ')).toBe(false)
    expect(isUsablePassword('1234')).toBe(true)
  })

  it('비밀번호를 저장하지 않는다', () => {
    const lock = makeLock('열려라참깨', '옛날 이야기')
    expect(lock).not.toBeNull()
    expect(JSON.stringify(lock)).not.toContain('열려라참깨')
  })

  it('같은 비밀번호라도 소금이 달라 해시가 갈린다', () => {
    const one = makeLock('1234')
    const two = makeLock('1234')
    expect(one?.salt).not.toBe(two?.salt)
    expect(one?.hash).not.toBe(two?.hash)
  })

  it('힌트는 선택이고, 안 적으면 키 자체가 없다', () => {
    expect(makeLock('1234')).not.toHaveProperty('hint')
    expect(makeLock('1234', '   ')).not.toHaveProperty('hint')
    expect(makeLock('1234', ' 생일 ')?.hint).toBe('생일')
  })
})

describe('비밀번호를 확인한다', () => {
  it('맞으면 연다', () => {
    const lock = makeLock('cjd1!')
    expect(lock && checkLock(lock, 'cjd1!')).toBe(true)
  })

  it('틀리면 안 연다', () => {
    const lock = makeLock('cjd1!')
    expect(lock && checkLock(lock, 'cjd1')).toBe(false)
    expect(lock && checkLock(lock, '')).toBe(false)
    expect(lock && checkLock(lock, 'CJD1!')).toBe(false)
  })

  it('한 프로젝트의 비밀번호로 다른 프로젝트를 열지 못한다 - 소금이 다르다', () => {
    const mine = makeLock('1234')
    const yours = makeLock('1234')
    expect(mine && yours && mine.hash).not.toBe(yours?.hash)
    // 소금을 섞으면 안 맞는다.
    expect(yours && checkLock({ ...yours, salt: mine?.salt ?? '' }, '1234')).toBe(false)
  })
})

describe('레코드에서 읽은 값이 잠금인가', () => {
  it('만든 것은 잠금이다', () => {
    expect(isLock(makeLock('1234', '생일'))).toBe(true)
    expect(isLock(makeLock('1234'))).toBe(true)
  })

  /**
   * **손상된 잠금으로 문을 닫지 않는다.** 닫으면 학생은 맞는 비밀번호를 넣고도 못
   * 들어가고 되살릴 길이 없다 - 잠기지 않은 것으로 치는 쪽이 덜 나쁘다.
   */
  it('모양이 아니면 잠기지 않은 것으로 친다', () => {
    expect(isLock(undefined)).toBe(false)
    expect(isLock(null)).toBe(false)
    expect(isLock('1234')).toBe(false)
    expect(isLock({})).toBe(false)
    expect(isLock({ salt: 'a' })).toBe(false)
    expect(isLock({ hash: 'b' })).toBe(false)
    expect(isLock({ salt: '', hash: 'b' })).toBe(false)
    expect(isLock({ salt: 'a', hash: '' })).toBe(false)
    expect(isLock({ salt: 'a', hash: 'b', hint: 3 })).toBe(false)
  })
})
