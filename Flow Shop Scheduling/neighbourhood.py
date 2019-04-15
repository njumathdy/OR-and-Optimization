# encoding: utf-8
import random
from itertools import combinations, permutations

import flow

##############################
# 在每个点处生成候选集
##############################

def neighbours_random(data, perm, num = 1):
    # 返回num个job permutations, 包括当前的job permutation
    candidates = [perm]
    for i in range(num):
        candidate = perm[:]
        random.shuffle(candidate)
        candidates.append(candidate)
    return candidates

def neighbours_swap(data, perm):
    # 交换工作的两两排列
    candidates = [perm]
    for (i,j) in combinations(range(len(perm)), 2):
        candidate = perm[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidates.append(candidate)
    return candidates

def neighbours_LNS(data, perm, size = 2):
    # Returns the Large Neighbourhood Search neighbours
    # 返回最大的相邻搜索点集
    candidates = [perm]

    # 限定最大的搜索数量
    neighbourhoods = list(combinations(range(len(perm)), size))
    random.shuffle(neighbourhoods)

    for subset in neighbourhoods[:flow.MAX_LNS_NEIGHBOURHOODS]:

        # 记录每个候选集中表现最好的候选点
        best_make = flow.makespan(data, perm)
        best_perm = perm

        # Enumerate every permutation of the selected neighbourhood
        # 枚举选中邻域的每个排列
        for ordering in permutations(subset):
            candidate = perm[:]
            for i in range(len(ordering)):
                candidate[subset[i]] = perm[ordering[i]]
            res = flow.makespan(data, candidate)
            if res < best_make:
                best_make = res
                best_perm = candidate

        # 将最好的结果保存到候选集尾部
        candidates.append(best_perm)

    return candidates

def neighbours_idle(data, perm, size=4):
    # 返回前size个空闲时间最长的jobs
    candidates = [perm]

    # 计算每个job的空闲时间
    sol = flow.compile_solution(data, perm)
    results = []

    for i in range(len(data)):
        finish_time = sol[-1][i] + data[perm[i]][-1]
        idle_time = (finish_time - sol[0][i]) - sum([t for t in data[perm[i]]])
        results.append((idle_time, i))

    subset = [job_ind for (idle, job_ind) in reversed(sorted(results))][:size]

    # 枚举这些空闲的工作的全排列
    for ordering in permutations(subset):
        candidate = perm[:]
        for i in range(len(ordering)):
            candidate[subset[i]] = perm[ordering[i]]
        candidates.append(candidate)

    return candidates