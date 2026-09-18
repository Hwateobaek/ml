/**
 * 2026-09-18에 들어온 모델 — 나무·숲·KNN의 회귀, 그레이디언트 부스팅, DBSCAN.
 *
 * **지키는 것은 셋이다.** 배우는가(쉬운 데이터에서 맞히는가), 저장했다 읽은 모델이 원본과
 * 같은 답을 내는가(해석기를 학습 쪽도 쓰는 구조가 실제로 그런가), 그리고 DBSCAN의 잡음이
 * 지표를 깨지 않는가.
 */

import { describe, expect, it } from 'vitest'

import { fit, type FitInput } from '../src/ml/engines/mljs'
import { fitDbscan } from '../src/ml/engines/dbscan'
import { baselineMs } from '../src/ml/estimate'
import { evaluateCluster } from '../src/ml/metrics'
import { loadModel, loadModelProba } from '../src/ml/models'

/** 계단 함수 — 나무가 정확히 맞혀야 하는 모양이다. */
function stepData(): { features: number[][]; target: number[] } {
  const features: number[][] = []
  const target: number[] = []
  for (let i = 0; i < 40; i += 1) {
    features.push([i, (i * 7) % 5])
    target.push(i < 20 ? 10 : 30)
  }
  return { features, target }
}

function input(
  features: number[][],
  target: (string | number)[],
  taskType: FitInput['taskType'],
  hyperparameters: Record<string, unknown> = {},
): FitInput {
  return {
    features,
    rowIndices: features.map((_, index) => index),
    target,
    taskType,
    hyperparameters,
    randomState: 42,
  }
}

describe('회귀로 넓힌 나무·숲·KNN', () => {
  for (const algorithm of ['decision_tree', 'random_forest', 'knn']) {
    it(`${algorithm} 회귀가 계단을 배우고, 저장했다 읽어도 같은 답이다`, async () => {
      const { features, target } = stepData()
      const trained = await fit(algorithm, input(features, target, 'regression'))
      const queries = [
        [3, 1],
        [35, 2],
      ]
      const answers = trained.predict(queries)
      expect(answers.every((value) => typeof value === 'number')).toBe(true)
      expect(Number(answers[0])).toBeLessThan(20)
      expect(Number(answers[1])).toBeGreaterThan(20)

      const reloaded = loadModel(trained.model, {
        trainingRows: {
          indices: features.map((_, index) => index),
          features,
          target: target.map(String),
        },
      })
      expect(reloaded(queries)).toEqual(answers)
    })
  }

  it('결정트리 회귀는 계단을 정확히 맞힌다', async () => {
    const { features, target } = stepData()
    const trained = await fit('decision_tree', input(features, target, 'regression'))
    expect(trained.predict(features)).toEqual(target)
  })

  it('랜덤 포레스트 회귀는 같은 씨앗이면 같은 숲이다', async () => {
    const { features, target } = stepData()
    const a = await fit('random_forest', input(features, target, 'regression'))
    const b = await fit('random_forest', input(features, target, 'regression'))
    expect(a.model).toEqual(b.model)
  })

  it('KNN 회귀는 이웃 k개의 평균이다', async () => {
    const features = [[0], [1], [2], [10]]
    const trained = await fit('knn', input(features, [1, 2, 3, 100], 'regression', { k: 3 }))
    expect(trained.predict([[0.5]])).toEqual([2])
  })

  it('분류는 그대로 분류다', async () => {
    const { features, target } = stepData()
    const labels = target.map((value) => (value > 20 ? 'big' : 'small'))
    const trained = await fit('decision_tree', input(features, labels, 'classification'))
    expect(trained.predict([[3, 1]])).toEqual(['small'])
  })
})

describe('그레이디언트 부스팅', () => {
  it('회귀가 오차를 줄이고, 저장했다 읽어도 같은 답이다', async () => {
    const { features, target } = stepData()
    const trained = await fit('gradient_boosting', input(features, target, 'regression'))
    const predicted = trained.predict(features).map(Number)
    const error = predicted.reduce((sum, value, i) => sum + Math.abs(value - (target[i] ?? 0)), 0)
    expect(error / predicted.length).toBeLessThan(0.1)
    expect(loadModel(trained.model)(features)).toEqual(trained.predict(features))
  })

  it('이진 분류를 맞히고 확률의 합이 1이다', async () => {
    const { features, target } = stepData()
    const labels = target.map((value) => (value > 20 ? 'yes' : 'no'))
    const trained = await fit('gradient_boosting', input(features, labels, 'classification'))
    expect(trained.predict(features)).toEqual(labels)
    const proba = loadModelProba(trained.model)
    expect(proba?.classes).toEqual(['no', 'yes'])
    const row = proba?.predict([[3, 1]])[0]
    expect((row?.[0] ?? 0) + (row?.[1] ?? 0)).toBeCloseTo(1, 12)
    expect(row?.[0] ?? 0).toBeGreaterThan(0.5)
  })

  it('다중 분류를 맞히고, 저장했다 읽어도 같은 답이다', async () => {
    const features: number[][] = []
    const labels: string[] = []
    for (let i = 0; i < 60; i += 1) {
      features.push([i % 3, (i * 13) % 7])
      labels.push(['a', 'b', 'c'][i % 3] as string)
    }
    const trained = await fit('gradient_boosting', input(features, labels, 'classification'))
    expect(trained.predict(features)).toEqual(labels)
    expect(loadModel(trained.model)(features)).toEqual(labels)
  })

  it('학습률을 낮추면 같은 그루 수에서 덜 맞힌다', async () => {
    const { features, target } = stepData()
    const loss = async (learningRate: number): Promise<number> => {
      const trained = await fit(
        'gradient_boosting',
        input(features, target, 'regression', { nEstimators: 5, learningRate }),
      )
      return trained
        .predict(features)
        .reduce<number>((sum, value, i) => sum + (Number(value) - (target[i] ?? 0)) ** 2, 0)
    }
    expect(await loss(0.01)).toBeGreaterThan(await loss(0.5))
  })
})

describe('DBSCAN', () => {
  /** 두 무리와 멀리 떨어진 점 하나. */
  const blobs: number[][] = [
    [0, 0],
    [0, 0.1],
    [0.1, 0],
    [0.1, 0.1],
    [5, 5],
    [5, 5.1],
    [5.1, 5],
    [5.1, 5.1],
    [20, 20],
  ]

  it('두 무리를 찾고 떨어진 점을 잡음(-1)으로 둔다', () => {
    const result = fitDbscan(blobs, 0.5, 3)
    expect(result.clusterCount).toBe(2)
    expect(Array.from(result.labels)).toEqual([0, 0, 0, 0, 1, 1, 1, 1, -1])
  })

  it('반경이 너무 작으면 전부 잡음이다', () => {
    const result = fitDbscan(blobs, 0.01, 3)
    expect(result.clusterCount).toBe(0)
    expect(Array.from(result.labels).every((label) => label === -1)).toBe(true)
  })

  it('학습 결과에 잡음 경고가 붙고, 새 점은 가까운 핵심 점의 무리다', async () => {
    const trained = await fit(
      'dbscan',
      input(
        blobs,
        blobs.map(() => ''),
        'clustering',
        { eps: 0.5, minSamples: 3 },
      ),
    )
    expect(trained.warning).toEqual({ code: 'DBSCAN_NOISE', params: { noise: 1, clusters: 2 } })
    expect(trained.clusterResult?.assignments).toEqual([0, 0, 0, 0, 1, 1, 1, 1, -1])
    const queries = [
      [0.05, 0.05],
      [5.05, 5.05],
      [10, 10],
    ]
    expect(trained.predict(queries)).toEqual(['0', '1', '-1'])
    expect(loadModel(trained.model)(queries)).toEqual(['0', '1', '-1'])
  })

  it('지표는 잡음을 빼고 낸다 — 잡음이 있어도 유한하다', async () => {
    const trained = await fit(
      'dbscan',
      input(
        blobs,
        blobs.map(() => ''),
        'clustering',
        { eps: 0.5, minSamples: 3 },
      ),
    )
    const cluster = trained.clusterResult
    expect(cluster).toBeDefined()
    const evaluation = evaluateCluster(blobs, cluster!.assignments, cluster!.centroids, 42)
    expect(evaluation.metrics.silhouette).toBeGreaterThan(0.9)
    expect(Number.isFinite(evaluation.metrics.inertia)).toBe(true)
  })

  it('전부 잡음이어도 지표가 던지지 않는다', () => {
    expect(() =>
      evaluateCluster(
        blobs,
        blobs.map(() => -1),
        [],
        42,
      ),
    ).not.toThrow()
  })
})

describe('예상 시간', () => {
  it('회귀는 회귀 표를 쓴다 — 분류 표보다 훨씬 짧다', () => {
    const base = {
      algorithm: 'decision_tree',
      dataType: 'tabular' as const,
      rows: 5000,
      columns: 8,
      hyperparameters: {},
    }
    const classification = baselineMs({ ...base, taskType: 'classification' }) ?? 0
    const regression = baselineMs({ ...base, taskType: 'regression' }) ?? 0
    expect(regression).toBeGreaterThan(0)
    expect(regression * 10).toBeLessThan(classification)
  })

  it('그레이디언트 부스팅은 그루 수에 선형이다', () => {
    const base = {
      algorithm: 'gradient_boosting',
      dataType: 'tabular' as const,
      rows: 1000,
      columns: 8,
    }
    const hundred = baselineMs({ ...base, hyperparameters: { nEstimators: 100 } }) ?? 0
    const two = baselineMs({ ...base, hyperparameters: { nEstimators: 200 } }) ?? 0
    expect(two / hundred).toBeCloseTo(2, 6)
  })
})
