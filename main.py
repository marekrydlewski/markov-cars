#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson, skellam
from collections import defaultdict

# | | | car rental 1
# |
# | car rental 2
rent_1 = 3
return_1 = 3
rent_2 = 4
returns_2 = 2
move_cost = 20
car_reward = 100
gamma = 0.9
M = 20
threshold = 12

# skellam distribution
rentals = [[skellam.pmf(x, 3, 3) for x in range(-20, 21, 1)], [skellam.pmf(x, 2, 4) for x in range(-20, 21, 1)]]

# poisson distribution
rents = [[poisson.pmf(x, mu=3) for x in range(21)], [poisson.pmf(x, mu=4) for x in range(21)]]
returns = [[poisson.pmf(x, mu=3) for x in range(21)], [poisson.pmf(x, mu=2) for x in range(21)]]

rents_cdf = [[poisson.cdf(x, mu=3) for x in range(21)], [poisson.cdf(x, mu=4) for x in range(21)]]
returns_cdf = [[poisson.cdf(x, mu=3) for x in range(21)], [poisson.cdf(x, mu=2) for x in range(21)]]


def get_particular_prob(i, i_poss, rent_num):
    if i_poss == M:
        partial_sum = 1.0
        for x in range(0, i + 1):
            partial_sum -= rents[rent_num][x] * (returns_cdf[rent_num][x + i_poss - i] - returns[rent_num][x + i_poss - i])
        partial_sum -= (1 - rents_cdf[rent_num][i]) * (returns_cdf[rent_num][i_poss])
        return partial_sum
    else:
        partial_sum = 0.0
        start = i - i_poss if i > i_poss else 0
        for x in range(start, i + 1):
            partial_sum += rents[rent_num][x] * returns[rent_num][x + i_poss - i]
        partial_sum += (1 - rents_cdf[rent_num][i]) * returns[rent_num][i_poss]
        return partial_sum


# prob maps
distributions_i = defaultdict(float)
distributions_j = defaultdict(float)


def get_distributions():
    for i in range(M + 1):
        for i_poss in range(M + 1):
            distributions_i[(i, i_poss)] = get_particular_prob(i, i_poss, 0)
            distributions_j[(i, i_poss)] = get_particular_prob(i, i_poss, 1)


def get_rentals_prob(rental, i):
    i_normalized = i + 20
    return rentals[rental][i_normalized]


# i and j before move
def get_reward(i, j, move):
    i_restrained, j_restrained = get_move_possible(i, j, move)
    if i_restrained < 0 or j_restrained < 0:
        print("Something went wrong")
        return 0

    r1 = 0.0
    for p1 in range(i_restrained + 1):
        r1 += (rents[0][p1] * p1)
    r1 += (1.0 - rents_cdf[0][i_restrained]) * i_restrained

    r2 = 0.0
    for p2 in range(j_restrained + 1):
        r2 += (rents[1][p2] * p2)
    r2 += (1.0 - rents_cdf[1][j_restrained]) * j_restrained

    return (r1 + r2) * 100 - abs(move) * 20  # + i + j


def get_discount(i, j, move, util):
    return get_reward(i, j, move) + gamma * get_conditional_prob_utils(i, j, move, util)


def get_conditional_prob_utils_skellam(i, j, move, util):
    i, j = get_move_possible(i, j, move)
    # normalization, count all for now
    local_sum = 0.0
    for i_poss, row in enumerate(util):
        for j_poss, item in enumerate(row):
            local_sum += get_rentals_prob(0, i_poss - i) * get_rentals_prob(1, j_poss - j)

    prob = 0.0
    for i_poss, row in enumerate(util):
        for j_poss, item in enumerate(row):
            prob += (get_rentals_prob(0, i_poss - i) * get_rentals_prob(1, j_poss - j)) / local_sum * util[i_poss][
                j_poss]
    return prob


def get_conditional_prob_utils(i, j, move, util):
    i, j = get_move_possible(i, j, move)
    partial_util = np.zeros((M + 1, M + 1)).tolist()

    for i_poss, row in enumerate(util):
        for j_poss, item in enumerate(row):
            partial_util[i_poss][j_poss] = distributions_i[i, i_poss] * distributions_j[j, j_poss]

    prob = 0.0
    for i_poss, row in enumerate(util):
        for j_poss, item in enumerate(row):
            prob += partial_util[i_poss][j_poss] * util[i_poss][j_poss]

    return prob


def get_bellman(i, j, util):
    moves_scores = []
    for move in range(-5, 6, 1):
        # check whether move is possible
        if i - move >= 0 and j + move >= 0:  # and i - move <= M and j + move <= M:
            moves_scores.append((move, get_discount(i, j, move, util)))
    return max(moves_scores, key=lambda p: p[1])


def get_move_possible(i, j, move):
    # check what if is too much cars in one rental
    i -= move
    j += move
    if i > M:
        i = M
    if j > M:
        j = M
    return i, j


def get_diff_utils(util1, util2):
    max_diff = 0.0
    for i in range(M + 1):
        for j in range(M + 1):
            if abs(util1[i][j] - util2[i][j]) > max_diff:
                max_diff = abs(util1[i][j] - util2[i][j])

    return max_diff


def value_iteration():
    utilities = np.zeros((M + 1, M + 1)).tolist()
    utilities_next = np.zeros((M + 1, M + 1)).tolist()
    # for r in range(45):
    # r = 0
    while True:
        utilities = utilities_next
        utilities_next = np.zeros((M + 1, M + 1)).tolist()
        # print(r)
        # r += 1
        for i, row in enumerate(utilities):
            for j, item in enumerate(row):
                utilities_next[i][j] = get_bellman(i, j, utilities)[1]

        diff = get_diff_utils(utilities, utilities_next)
        if diff <= threshold:
            break
    return utilities_next


def get_policy(utils):
    # it has to have the same transition model like in get_conditional_prob
    policy = np.zeros((M + 1, M + 1)).tolist()
    for i, row in enumerate(utils):
        for j, item in enumerate(row):
            policy[i][j] = get_policy_for_field(i, j, utils)
    return policy


def get_policy_for_field(i, j, utils):
    moves_scores = []
    for move in range(-5, 6, 1):
        # check whether move is possible
        if i - move >= 0 and j + move >= 0:  # and i - move <= M and j + move <= M:
            moves_scores.append((move, get_discount(i, j, move, utils)))
    return max(moves_scores, key=lambda p: p[1])[0]


if __name__ == "__main__":

    get_distributions()
    utils = value_iteration()

    # for i in range(M + 1):
    #     for j in range(M + 1):
    #         print('{:8.2f}'.format(utils[i][j]), end=' ')
    #     print()

    policy = get_policy(utils)

    for i in range(M + 1):
        for j in range(M + 1):
            print("{:2d} ".format(policy[i][j]), end='')
        print()
