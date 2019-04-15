# encoding: utf-8
import sys, os, time, random

from functools import partial
from collections import namedtuple
from itertools import product

import neighbourhood as neigh
import heuristics as heur

##############
# 各类参数设置
##############
TIME_LIMIT = 300.0 # 总运行时间
TIME_INCREMENT = 13.0 # 评价各种启发式搜索策略的间隔时间
DEBUG_SWITCH = False # Debug设置
MAX_LNS_NEIGHBOURHOODS = 1000 # 最大的neighbourboods数量


################
# 策略
################
# 策略是指我们如何产生每个点的neighbourhood集合以及如何用启发式的方法确定下一个可行点


STRATEGIES = []

Strategy = namedtuple('Strategy', ['name', 'neighbourhood', 'heuristic'])

def initialize_strategies():

    global STRATEGIES

    # 可能用到的各种邻域选择方法
    neighbourhoods = [
        ('Random Permutation', partial(neigh.neighbours_random, num=100)),
        ('Swapped Pairs', neigh.neighbours_swap),
        ('Large Neighbourhood Search (2)', partial(neigh.neighbours_LNS, size=2)),
        ('Large Neighbourhood Search (3)', partial(neigh.neighbours_LNS, size=3)),
        ('Idle Neighbourhood (3)', partial(neigh.neighbours_idle, size=3)),
        ('Idle Neighbourhood (4)', partial(neigh.neighbours_idle, size=4)),
        ('Idle Neighbourhood (5)', partial(neigh.neighbours_idle, size=5))
    ]

    # 确定下一个可行点的启发式方法
    heuristics = [
        ('Hill Climbing', heur.heur_hillclimbing),
        ('Random Selection', heur.heur_random),
        ('Biased Random Selection', heur.heur_random_hillclimbing)
    ]

    # 邻域选择方法和可行点确定方法的组合
    for (n, h) in product(neighbourhoods, heuristics):
        STRATEGIES.append(Strategy("%s / %s" % (n[0], h[0]), n[1], h[1]))


# 主函数，求解flow shop scheduling问题
def solve(data):

    # 初始化
    initialize_strategies()
    global STRATEGIES

    # 每中策略的相关信息:
    #  improvements: 该策略对解的提高的程度
    #  time_spent: 该策略所耗费的时间
    #  weights: 策略的权重
    #  usage: 使用该策略的次数
    strat_improvements = {strategy: 0 for strategy in STRATEGIES}
    strat_time_spent = {strategy: 0 for strategy in STRATEGIES}
    strat_weights = {strategy: 1 for strategy in STRATEGIES}
    strat_usage = {strategy: 0 for strategy in STRATEGIES}

    # 随机初始化
    perm = range(len(data))
    random.shuffle(perm)

    # 记录最优解
    best_make = makespan(data, perm)
    best_perm = perm
    res = best_make

    # 每次迭代的相关信息
    iteration = 0
    time_limit = time.time() + TIME_LIMIT
    time_last_switch = time.time()

    time_delta = TIME_LIMIT / 10
    checkpoint = time.time() + time_delta
    percent_complete = 10

    print("\nSolving...")

    while time.time() < time_limit:

        if time.time() > checkpoint:
            print(" %d %%" % percent_complete)
            percent_complete += 10
            checkpoint += time_delta

        iteration += 1

        # 启发式地选取最优策略
        strategy = pick_strategy(STRATEGIES, strat_weights)

        old_val = res
        old_time = time.time()

        # 利用当前策略在该策略确定的候选集中选取下一个可行点
        candidates = strategy.neighbourhood(data, perm)
        perm = strategy.heuristic(data, candidates)
        res = makespan(data, perm)

        # 更新该策略的统计量
        strat_improvements[strategy] += res - old_val # 可能为负
        strat_time_spent[strategy] += time.time() - old_time
        strat_usage[strategy] += 1

        if res < best_make:
            best_make = res
            best_perm = perm[:]

        # 更新对策略的评价，赋予那些更为有效的策略更大的权重
        if time.time() > time_last_switch + TIME_INCREMENT:

            # Normalize the improvements made by the time it takes to make them
            results = sorted([(float(strat_improvements[s]) / max(0.001, strat_time_spent[s]), s)
                              for s in STRATEGIES])

            if DEBUG_SWITCH:
                print("\nComputing another switch...")
                print("Best performer: %s (%d)" % (results[0][1].name, results[0][0]))
                print("Worst performer: %s (%d)" % (results[-1][1].name, results[-1][0]))

            # 对于较为优秀的策略，提高其权重
            for i in range(len(STRATEGIES)):
                strat_weights[results[i][1]] += len(STRATEGIES) - i

                # 对于没有被使用的权重，也提高其权重防止在后续的搜索中被遗忘
                if results[i][0] == 0:
                    strat_weights[results[i][1]] += len(STRATEGIES)

            time_last_switch = time.time()

            if DEBUG_SWITCH:
                print(results)
                print(sorted([strat_weights[STRATEGIES[i]] for i in range(len(STRATEGIES))]))

            # 一轮迭代结束，将与该论迭代相关的信息重置
            strat_improvements = {strategy: 0 for strategy in STRATEGIES}
            strat_time_spent = {strategy: 0 for strategy in STRATEGIES}


    print(" %d %%\n" % percent_complete)
    print("\nWent through %d iterations." % iteration)

    print("\n(usage) Strategy:")
    results = sorted([(strat_weights[STRATEGIES[i]], i)
                      for i in range(len(STRATEGIES))], reverse=True)
    for (w, i) in results:
        print("(%d) \t%s" % (strat_usage[STRATEGIES[i]], STRATEGIES[i].name))

    return (best_perm, best_make)


# 读取问题数据
def parse_problem(filename, k=1):
    # Taillard problem file: flow shop scheduling标准数据集
    # http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/ordonnancement.html

    print("\nParsing...")

    with open(filename, 'r') as f:
        problem_line = '/number of jobs, number of machines, initial seed, upper bound and lower bound :/'

        lines = list(map(str.strip, f.readlines()))

        lines[0] = '/' + lines[0]

        try:
            lines = '/'.join(lines).split(problem_line)[k].split('/')[2:]
        except IndexError:
            max_instances = len('/'.join(lines).split(problem_line)) - 1
            print("\nError: Instance must be within 1 and %d\n" % max_instances)
            sys.exit(0)

        # 转化为整数
        data = [map(int, line.split()) for line in lines]

    # 对data做转置，使得data中每个元素表示每个job需要的时间
    return list(zip(*data))


def pick_strategy(strategies, weights):
    # 随机选取一个策略，并非完全随机，而是采用轮盘赌的方式
    total = sum([weights[strategy] for strategy in strategies])
    pick = random.uniform(0, total)
    count = weights[strategies[0]]

    i = 0
    while pick > count:
        count += weights[strategies[i+1]]
        i += 1

    return strategies[i]


def makespan(data, perm):
    # 计算当前解需要的总时间：最迟的完成时间减去最早的开始时间。
    return compile_solution(data, perm)[-1][-1] + data[perm[-1]][-1]


def compile_solution(data, perm):
    # 给定工作安排，计算每个机器的时间表

    num_machines = len(data[0])

    machine_times = [[] for _ in range(num_machines)]

    # 将第一个job分配给各个machine
    machine_times[0].append(0)
    for mach in range(1,num_machines):
        machine_times[mach].append(machine_times[mach-1][0] + data[perm[0]][mach-1])

    # 分配剩下的jobs
    for i in range(1, len(perm)):

        # 第一台机器从来不会有空闲时间
        # job = perm[i]
        machine_times[0].append(machine_times[0][-1] + data[perm[i-1]][0])

        # 对于其余的机器，其开始时间是max(该job的前一个task完成的时间，该机器完成前一个job的task的时间)
        for mach in range(1, num_machines):
            machine_times[mach].append(max(machine_times[mach-1][i] + data[perm[i]][mach-1],
                                        machine_times[mach][i-1] + data[perm[i-1]][mach]))

    return machine_times


# 输出结果的相关信息
def print_solution(data, perm):

    sol = compile_solution(data, perm)

    print("\nPermutation: %s\n" % str([i+1 for i in perm]))

    print("Makespan: %d\n" % makespan(data, perm))

    row_format ="{:>15}" * 4
    print(row_format.format('Machine', 'Start Time', 'Finish Time', 'Idle Time'))
    for mach in range(len(data[0])):
        finish_time = sol[mach][-1] + data[perm[-1]][mach]
        idle_time = (finish_time - sol[mach][0]) - sum([job[mach] for job in data]) # 该台机器的空闲时间
        print(row_format.format(mach+1, sol[mach][0], finish_time, idle_time))

    results = []
    for i in range(len(data)):
        finish_time = sol[-1][i] + data[perm[i]][-1]
        idle_time = (finish_time - sol[0][i]) - sum([time for time in data[perm[i]]])
        results.append((perm[i]+1, sol[0][i], finish_time, idle_time))

    print("\n")
    print(row_format.format('Job', 'Start Time', 'Finish Time', 'Idle Time'))
    for r in sorted(results):
        print(row_format.format(*r))

    print("\n\nNote: Idle time does not include initial or final wait time.\n")


if __name__ == '__main__':

    if len(sys.argv) == 2:
        data = parse_problem(sys.argv[1])
    elif len(sys.argv) == 3:
        data = parse_problem(sys.argv[1], int(sys.argv[2]))
    else:
        print("\nUsage: python flow.py <Taillard problem file> [<instance number>]\n")
        sys.exit(0)

    (perm, ms) = solve(data)
    print_solution(data, perm)