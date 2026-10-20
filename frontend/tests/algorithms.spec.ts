/**
 * 알고리즘 등록부와 선택 가능 판정.
 *
 * 여기서 지켜야 하는 것 둘.
 *
 *   1. **분기가 없다.** 등록부에 항목을 추가하면 화면이 따라온다. 회귀·군집이 들어와도
 *      `if (taskType === ...)` 가 생기면 안 된다 (mlpx-spec.md 0.1)
 *   2. **못 쓰는 것에도 이유가 있다.** 그리고 그 이유가 학생에게 쓸모 있어야 한다 -
 *      이미지 데이터에 회귀를 고른 학생에게 "서버가 없습니다"는 도움이 안 된다
 */

import { readFileSync } from 'node:fs'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { CLIENT_ERROR_CODES } from '../src/errors'
import { MAX_DATASET_ROWS, MAX_IMAGE_COUNT, MLJS_DECISION_TREE_ROW_LIMIT } from '../src/limits'
import { NOT_FOR_TABULAR_ALGORITHM, SKLEARN_ONLY_ALGORITHM } from './fixtures/algorithms'
import {
  ALGORITHMS,
  algorithmOptions,
  enabledAlgorithms,
  type Algorithm,
  type Selection,
} from '../src/ml/algorithms'
import {
  BROWSER_RUNTIME_IDS,
  RUNTIME_IDS,
  UNMEASURED,
  UNMEASURED_BASELINE,
  type EngineState,
  type RuntimeContext,
} from '../src/ml/backend'
import { DATA_TYPES, TASK_TYPES, type DataType } from '../src/project/schema'

const tabularClassification: Selection = { dataType: 'tabular', taskType: 'classification' }

function context(overrides: Partial<RuntimeContext> = {}): RuntimeContext {
  return {
    serverStatus: 'unavailable',
    rowCount: 100,
    dataType: 'tabular',
    limitsOff: false,
    ...overrides,
  }
}

const skReady: Record<string, EngineState> = { 'pyodide-sklearn': 'ready' }

function optionFor(options: ReturnType<typeof algorithmOptions>, id: string) {
  return options.find((option) => option.algorithm.id === id)
}

describe('등록부', () => {
  it('id가 겹치지 않는다', () => {
    expect(new Set(ALGORITHMS.map((a) => a.id)).size).toBe(ALGORITHMS.length)
  })

  it('아무 데서도 안 서는 줄이 없다', () => {
    // **어휘는 이제 타입이 지킨다** (ml/axes.ts) - 모르는 축 값을 적거나 아는 값을
    // 빠뜨리면 컴파일이 깨진다. 그래서 여기서 어휘를 다시 세지 않는다.
    //
    // 타입이 못 잡는 것은 **칸을 다 채웠는데 전부 false인 줄**이다. 화면 어디에도
    // 안 나오는데 등록부에는 있는 항목이고, 그건 지웠어야 할 줄이다 (§9.2.1).
    for (const algorithm of ALGORITHMS) {
      expect(
        DATA_TYPES.some((dataType) => algorithm.dataTypes[dataType]),
        algorithm.id,
      ).toBe(true)
      expect(
        TASK_TYPES.some((taskType) => algorithm.taskTypes[taskType]),
        algorithm.id,
      ).toBe(true)
      expect(
        RUNTIME_IDS.some((runtimeId) => algorithm.runtimes[runtimeId]),
        algorithm.id,
      ).toBe(true)
    }
  })

  it('그 알고리즘을 돌리는 엔진의 칸은 재 본 값이다', () => {
    // **전역 기본값에 얹혀 있는 줄이 없어야 한다.** 일곱 줄 전부 실측이 있고
    // (open-decisions.md #13), 비워 두면 다음에 전역을 올리는 사람이 그 알고리즘까지
    // 대신 허락하게 된다. 값이 안 바뀌는 랜덤포레스트도 그래서 적혀 있다.
    //
    // **다른 칸을 여기서 요구하지 않는다** - `pyodide-sklearn`은 아직 아무것도 안 쟀고,
    // `UNMEASURED`가 그 사실을 적은 것이다. 지어낸 숫자보다 낫다.
    //
    // **다루는 종류의 칸만 요구한다** - 등록부가 닫아 둔 칸(회귀의 이미지)은 판정이
    // 닿지 않는 자리라 지어낸 숫자보다 `UNMEASURED`가 정직하다.
    //
    // **`mljs`를 이름으로 묻지 않는다** (2026-10-20). 그 엔진에서 아예 안 도는
    // 알고리즘이 들어오면서(`cnn`은 `tfjs` 하나뿐이다) **있어서는 안 되는 상수를
    // 내놓으라고 조르는 검사**가 됐다. 묻는 것은 그대로다 — *"이 줄이 전역 기본값에
    // 얹혀 있지 않은가"*이고, 그 답은 **돌릴 수 있는 엔진 중 하나라도 재어졌는가**다.
    for (const algorithm of ALGORITHMS) {
      for (const dataType of DATA_TYPES) {
        if (!algorithm.dataTypes[dataType]) continue
        const measured = BROWSER_RUNTIME_IDS.filter(
          (runtimeId) =>
            algorithm.runtimes[runtimeId] &&
            typeof algorithm.maxRows[dataType][runtimeId] === 'number',
        )
        expect(measured, `${algorithm.id}/${dataType}`).not.toHaveLength(0)
      }
    }
  })

  it('행 상한이 그 종류의 데이터 상한을 넘지 않는다', () => {
    // 넘는 값은 거짓말이다 - 그만큼의 행은 애초에 앱에 들어오지 못한다.
    // **천장이 종류마다 다르다**: 표는 `MAX_DATASET_ROWS`, 이미지는 `MAX_IMAGE_COUNT`가
    // 업로드에서 이미 막는다 (limits.ts).
    const ceiling: Record<DataType, number> = { tabular: MAX_DATASET_ROWS, image: MAX_IMAGE_COUNT }
    for (const algorithm of ALGORITHMS) {
      for (const dataType of DATA_TYPES) {
        for (const runtimeId of BROWSER_RUNTIME_IDS) {
          const limit = algorithm.maxRows[dataType][runtimeId]
          if (limit === UNMEASURED) continue
          expect(limit, `${algorithm.id}/${dataType}/${runtimeId}`).toBeLessThanOrEqual(
            ceiling[dataType],
          )
        }
      }
    }
  })

  it('이미지에서 무거운 것은 이미지 칸이 더 낮다', () => {
    // **트리 계열 둘만 갈린다** (open-decisions.md #13). 갈린 값이 같은 상수를 가리키게
    // 되면 이 검사가 운다 - 그때 봐야 할 것은 검사가 아니라 상수다.
    for (const id of ['decision_tree', 'random_forest']) {
      const algorithm = ALGORITHMS.find((candidate) => candidate.id === id)
      const image = algorithm?.maxRows.image.mljs
      const tabular = algorithm?.maxRows.tabular.mljs
      expect(typeof image === 'number' && typeof tabular === 'number' && image < tabular, id).toBe(
        true,
      )
    }
  })

  it('순수 JS 구현이 없는 것도 숨기지 않고 등록한다', () => {
    // 목록에서 빼면 학생은 그런 모델이 있다는 사실조차 모른다. **표본은 가짜다** -
    // 등록부에는 지금 sklearn 전용이 하나도 없고, 그건 규칙이 아니라 오늘의 사실이다.
    const options = algorithmOptions(tabularClassification, context(), [SKLEARN_ONLY_ALGORITHM])
    expect(options).toHaveLength(1)
    expect(options[0]?.enabled).toBe(false)
    expect(options[0]?.reason).toBe('ENGINE_NOT_WIRED')
  })
})

describe('세 축으로 고른다', () => {
  it('과제 유형이 다르면 잠긴다 - 분류를 골랐는데 회귀 모델이 열리면 안 된다', () => {
    const options = algorithmOptions(tabularClassification, context())
    expect(optionFor(options, 'linear_regression')?.reason).toBe('ALGORITHM_NOT_FOR_TASK_TYPE')
  })

  it('회귀를 고르면 회귀 모델이 열리고 분류 모델이 잠긴다', () => {
    const options = algorithmOptions({ dataType: 'tabular', taskType: 'regression' }, context())
    expect(optionFor(options, 'linear_regression')?.enabled).toBe(true)
    expect(optionFor(options, 'svm')?.reason).toBe('ALGORITHM_NOT_FOR_TASK_TYPE')
  })

  it('데이터 타입이 다르면 잠긴다', () => {
    // **표본은 가짜다.** 어휘에는 지금 되는 종류만 있으므로(open-decisions.md) 안 맞는
    // 종류를 넘겨서 확인할 수 없다. 규칙을 확인하는 것이지 어휘를 확인하는 게 아니다.
    const options = algorithmOptions(tabularClassification, context(), [NOT_FOR_TABULAR_ALGORITHM])
    expect(options.every((option) => option.reason === 'ALGORITHM_NOT_FOR_DATA_TYPE')).toBe(true)
  })

  it('데이터 타입이 과제 유형보다 먼저다 - 더 근본적인 것이 먼저 걸린다', () => {
    // 이 모델은 표에서도 안 서고 회귀도 아니다. 둘 다 안 맞지만 학생이 먼저 알아야
    // 할 것은 데이터다.
    const options = algorithmOptions({ dataType: 'tabular', taskType: 'regression' }, context(), [
      NOT_FOR_TABULAR_ALGORITHM,
    ])
    expect(options[0]?.reason).toBe('ALGORITHM_NOT_FOR_DATA_TYPE')
  })

  it('축이 다 맞으면 실행 방법을 본다', () => {
    const options = algorithmOptions(tabularClassification, context())
    // 순수 JS가 있으므로 서버가 없어도 열린다.
    expect(optionFor(options, 'decision_tree')?.enabled).toBe(true)
    // sklearn에서만 도는 모델은 서버도 없고 엔진도 준비 안 됐으니 잠긴다.
    const sklearnOnly = algorithmOptions(tabularClassification, context(), [SKLEARN_ONLY_ALGORITHM])
    expect(sklearnOnly[0]?.enabled).toBe(false)
  })
})

describe('못 쓰는 이유가 쓸모 있어야 한다', () => {
  it('지원하지도 않는 실행 방법의 이유를 보여주지 않는다', () => {
    // 이 모델은 mljs를 아예 지원하지 않는다. "여기서 실행할 수 없습니다"라고만 하면
    // 학생은 무엇을 해야 하는지 모른다. **엔진 쪽 사유까지 내려가야 한다** - 지금은
    // 켤 자리가 없어서 그 사유가 `ENGINE_NOT_WIRED`이고, 배선이 붙으면
    // `ENGINE_NOT_READY`("준비하면 된다")로 돌아온다.
    const options = algorithmOptions(tabularClassification, context(), [SKLEARN_ONLY_ALGORITHM])
    expect(options[0]?.reason).toBe('ENGINE_NOT_WIRED')
  })

  it('엔진을 준비하면 sklearn 전용 모델이 열린다', () => {
    const options = algorithmOptions(tabularClassification, context({ engineStates: skReady }), [
      SKLEARN_ONLY_ALGORITHM,
    ])
    expect(options[0]?.enabled).toBe(true)
  })

  it('데이터가 너무 크면 브라우저 전용 상황에서 잠긴다', () => {
    // **알고리즘의 상한을 쓴다.** 전역 기본값으로 재면 상한이 그보다 높은 알고리즘에서
    // 이 검사가 아무것도 확인하지 않게 된다.
    const options = algorithmOptions(
      tabularClassification,
      context({ rowCount: MLJS_DECISION_TREE_ROW_LIMIT + 1 }),
    )
    expect(optionFor(options, 'decision_tree')?.reason).toBe('DATASET_TOO_LARGE_FOR_BROWSER')
  })

  it('서버가 있으면 큰 데이터도 열린다', () => {
    const options = algorithmOptions(
      tabularClassification,
      context({
        serverStatus: 'available',
        rowCount: MLJS_DECISION_TREE_ROW_LIMIT + 1,
        dataType: 'tabular',
      }),
    )
    expect(optionFor(options, 'decision_tree')?.enabled).toBe(true)
  })

  it('잠긴 항목에는 언제나 이유가 있고 그 이유가 로케일에 있다', () => {
    const cases = [
      algorithmOptions(tabularClassification, context()),
      algorithmOptions(tabularClassification, context(), [NOT_FOR_TABULAR_ALGORITHM]),
      algorithmOptions(tabularClassification, context({ rowCount: 999999 })),
      algorithmOptions({ dataType: 'tabular', taskType: 'regression' }, context()),
    ]
    const declared = new Set<string>(CLIENT_ERROR_CODES)
    for (const options of cases) {
      for (const option of options) {
        if (option.enabled) continue
        expect(option.reason, option.algorithm.id).toBeDefined()
        expect(declared.has(option.reason as string), option.reason).toBe(true)
      }
    }
  })

  it('잠겨 있어도 실행 방법별 판정을 함께 준다 - 무엇을 하면 되는지 알아야 한다', () => {
    const option = algorithmOptions(tabularClassification, context(), [SKLEARN_ONLY_ALGORITHM])[0]
    // **실행 방법 수를 세어 둔다.** 숫자를 박아 두면 실행 방법이 늘어날 때마다 여기가
    // 울고, 그때 고치는 사람이 "판정이 전부 왔는가"가 아니라 숫자를 본다.
    expect(option?.runtimes).toHaveLength(RUNTIME_IDS.length)
    expect(option?.runtimes.find((r) => r.runtime.id === 'mljs')?.reason).toBe(
      'ALGORITHM_NOT_AVAILABLE_HERE',
    )
    expect(option?.runtimes.find((r) => r.runtime.id === 'server-sklearn')?.reason).toBe(
      'SERVER_UNAVAILABLE',
    )
  })
})

describe('분기 없이 늘어난다', () => {
  it('군집에서는 k_means와 dbscan만 살고 나머지는 이유와 함께 잠긴다', () => {
    // **예전 이름은 "등록부에 없는 과제 유형을 골라도 전부 잠긴다"였다.** 그 검사는
    // 군집에 알고리즘이 하나도 없던 시절의 것이고, V3에서 k_means가 살면서 확인
    // 대상을 잃었다. 축의 한 값에서 전부 잠기는 상태는 `DataType`에 이미지가 들어오는
    // V4에서 다시 생긴다 - 그때 그 검사를 쓴다.
    const options = algorithmOptions({ dataType: 'tabular', taskType: 'clustering' }, context())
    expect(options).toHaveLength(ALGORITHMS.length)
    const enabled = enabledAlgorithms(options)
    expect(enabled.map((a) => a.id)).toEqual(['k_means', 'dbscan'])
    /**
     * **사유의 우선순위가 여기서 보인다** (`ml/algorithms.ts`의 머리말): 데이터 종류가
     * 과제 유형보다 먼저다. 표를 아예 안 하는 알고리즘(`cnn`)은 표 프로젝트에서
     * *"이 유형에 없다"*가 아니라 *"이 데이터 종류에 없다"*라고 말해야 한다 —
     * 합성곱을 쓰려는 학생이 할 일은 유형을 바꾸는 것이 아니라 사진을 올리는 것이다.
     */
    expect(
      options
        .filter((o) => !['k_means', 'dbscan'].includes(o.algorithm.id))
        .map((o) => [o.algorithm.id, o.reason] as const)
        .filter(([id, reason]) =>
          ALGORITHMS.find((a) => a.id === id)?.dataTypes.tabular
            ? reason !== 'ALGORITHM_NOT_FOR_TASK_TYPE'
            : reason !== 'ALGORITHM_NOT_FOR_DATA_TYPE',
        ),
    ).toEqual([])
  })

  it('등록부를 넘기면 그것만 본다 - 새 항목이 코드 변경 없이 들어온다', () => {
    const future: Algorithm[] = [
      {
        id: 'kmeans',
        dataTypes: { tabular: true, image: true },
        taskTypes: { classification: false, regression: false, clustering: true },
        featureSource: 'embedding',
        runtimes: { mljs: true, 'pyodide-sklearn': false, 'server-sklearn': false, tfjs: false },
        // 아직 아무도 안 쟀다. 군집을 실제로 넣는 날 이 칸이 숫자를 요구한다.
        maxRows: {
          tabular: { mljs: UNMEASURED, 'pyodide-sklearn': UNMEASURED, tfjs: UNMEASURED },
          image: { mljs: UNMEASURED, 'pyodide-sklearn': UNMEASURED, tfjs: UNMEASURED },
        },
        baseline: { tabular: UNMEASURED_BASELINE, image: UNMEASURED_BASELINE },
      },
    ]
    const options = algorithmOptions(
      { dataType: 'tabular', taskType: 'clustering' },
      context(),
      future,
    )
    expect(options[0]?.enabled).toBe(true)
  })
})

describe('enabledAlgorithms', () => {
  it('고를 수 있는 것만 남긴다', () => {
    const options = algorithmOptions(tabularClassification, context())
    const ids = enabledAlgorithms(options).map((a) => a.id)
    // 분류 모델 전부다. **회귀 전용만 빠진다** - 이제 sklearn 전용은 하나도 없다.
    expect(ids).toEqual([
      'decision_tree',
      'knn',
      'logistic_regression',
      'random_forest',
      'naive_bayes',
      'svm',
      'neural_network',
      'gradient_boosting',
    ])
  })
})

/**
 * **어느 상수가 어느 칸에 오는가.**
 *
 * 기존 검사 셋은 값의 *성질*만 본다 — 숫자인가, 데이터셋 천장 이하인가, 이미지 칸이 표
 * 칸보다 낮은가. 그래서 `decision_tree`와 `random_forest`의 이미지 칸을 **통째로
 * 맞바꿔도** 저장소 전체도 `vue-tsc`도 조용했다 (R13-3 감사 A-1). 맞바꾸면 사진
 * 1,000장에 랜덤포레스트가 열리는데 `limits.ts`의 실측이 그 자리를 521.7초라고 적는다.
 *
 * **값을 `limits.ts`에서 가져와 견주면 안 된다** — 자기 자신과 대조하는 것이라 맞바꿈을
 * 여전히 못 본다. 그래서 **소스를 글자로 읽어 상수의 이름**이 알고리즘 id와 맞는지 본다.
 */
describe('행 상한 칸에 제 이름의 상수가 온다', () => {
  const SOURCE = readFileSync(join(process.cwd(), 'src', 'ml', 'algorithms.ts'), 'utf-8')

  /**
   * 알고리즘 id -> (데이터 종류 -> 실행 방법 -> 상수 이름). 소스에서 글자로 읽는다.
   *
   * **실행 방법마다 읽는다** (2026-10-20). 전에는 `mljs` 칸만 봤는데, 그 엔진에서
   * 아예 안 도는 알고리즘이 들어오면서(`cnn`은 `tfjs` 하나뿐이다) **그 칸이 영영
   * `UNMEASURED`**가 됐다 — 검사가 "`MLJS_IMAGE_CNN_ROW_LIMIT`이 없다"고 우는데
   * 그런 상수는 있어서도 안 되는 것이다.
   *
   * **줄바꿈을 견딘다.** 칸이 셋이 되면서 prettier가 줄을 접었고, 한 줄짜리 정규식이
   * 알고리즘 **전부**를 "(없다)"로 읽었다 — 그때도 검사는 울었지만, 이렇게 **쉽게
   * 눈먼다**는 것이 이 자리의 성질이라 아래 "읽을 칸을 실제로 찾는다"가 함께 선다.
   */
  function cells(): Map<string, Record<string, Record<string, string | undefined>>> {
    const found = new Map<string, Record<string, Record<string, string | undefined>>>()
    // **id로 자른다.** 한 조각이 알고리즘 하나이므로 조각 안에서 찾은 칸은 그 알고리즘의
    // 것이다 - 블록의 끝을 정규식으로 맞히려 들면 들여쓰기에 매달린다.
    const chunks = SOURCE.split(/id: '/).slice(1)
    for (const chunk of chunks) {
      const id = /^([a-z_]+)'/.exec(chunk)?.[1]
      if (id === undefined) continue
      const perType: Record<string, Record<string, string | undefined>> = {}
      for (const dataType of DATA_TYPES) {
        // 그 종류의 칸 하나를 통째로 집는다. `s` 플래그라 접힌 줄도 한 덩어리다.
        const block = new RegExp(`${dataType}: \\{(.*?)\\}`, 's').exec(chunk)?.[1] ?? ''
        perType[dataType] = {
          mljs: /\bmljs: (\w+)/.exec(block)?.[1],
          tfjs: /\btfjs: (\w+)/.exec(block)?.[1],
        }
      }
      found.set(id, perType)
    }
    return found
  }

  /** 이 알고리즘이 그 실행 방법에서 실제로 도는가. 안 돌면 상한도 없어야 한다. */
  const RUNTIME_PREFIX: Readonly<Record<string, string>> = { mljs: 'MLJS_', tfjs: 'TFJS_' }

  /** `k_means` -> `KMEANS`, `decision_tree` -> `DECISIONTREE`. 밑줄은 이름마다 달라 접는다. */
  const core = (text: string) => text.toUpperCase().replace(/_/g, '')

  it('읽을 칸을 실제로 찾는다', () => {
    // 정규식이 썩으면 0개가 되고 아래 검사가 영원히 초록이 된다.
    expect(cells().size).toBe(ALGORITHMS.length)
  })

  for (const dataType of DATA_TYPES) {
    it(`${dataType} 칸의 상수 이름이 알고리즘과 맞는다`, () => {
      const wrong: string[] = []
      for (const algorithm of ALGORITHMS) {
        if (!algorithm.dataTypes[dataType]) continue
        for (const [runtime, prefix] of Object.entries(RUNTIME_PREFIX)) {
          // 그 엔진에 이 알고리즘 구현이 없으면 상한도 없다 — `UNMEASURED`가 정답이다.
          if (!algorithm.runtimes[runtime as keyof typeof algorithm.runtimes]) continue
          const name = cells().get(algorithm.id)?.[dataType]?.[runtime]
          // **안 잰 칸은 여기서 안 본다.** 이름이 없는 것이 정상이고, 재 보라고 미는
          // 일은 `estimate.spec.ts`의 "사진을 여는 알고리즘마다 사진 상한이 있다"가 한다.
          if (name === 'UNMEASURED') continue
          const expected = core(
            `${prefix}${dataType === 'image' ? 'IMAGE_' : ''}${algorithm.id}_ROW_LIMIT`,
          )
          if (name === undefined || core(name) !== expected) {
            wrong.push(`${algorithm.id}.${dataType}.${runtime} = ${name ?? '(없다)'}`)
          }
        }
      }
      expect(wrong, 'swapping these starts training at a size the student cannot run').toEqual([])
    })
  }
})
