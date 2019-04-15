# encoding: utf-8
import random

import flow

################
# 启发式搜索 
################

################
# 我们通过三种方案从候选集中选取下一个可行点

def heur_hillclimbing(data, candidates):
    # 返回候选集中最好的方案
    scores = [(flow.makespan(data, perm), perm) for perm in candidates]
    return sorted(scores)[0][1]

def heur_random(data, candidates):
    # 随机选取
    return random.choice(candidates)

def heur_random_hillclimbing(data, candidates):
    # 考虑每个方案的排名，进行加权随机选取
    scores = [(flow.makespan(data, perm), perm) for perm in candidates]
    i = 0
    while (random.random() < 0.5) and (i < len(scores) - 1):
        i += 1
    return sorted(scores)[i][1]