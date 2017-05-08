#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson, skellam

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


# skellam distribution
rentals = [[skellam.pmf(x, 3, 3) for x in range(-20, 21, 1)], [skellam.pmf(x, 2, 4) for x in range(-20, 21, 1)]]

# poisson distribution
rents = [[poisson.pmf(x, mu=3) for x in range(50)], [poisson.pmf(x, mu=4) for x in range(50)]]
returns = [[poisson.pmf(x, mu=3) for x in range(50)], [poisson.pmf(x, mu=2) for x in range(50)]]

# rewards via poisson distribution
rent_rewards = [[car_reward * x for x in rents[0]], [car_reward * x for x in rents[1]]]


def get_rentals_prob(rental, i):
    i_normalized = i + 20
    return rentals[rental][i_normalized]


# i and j before move
def get_reward(i, j, move):
    i, j = get_move_possible(i, j, move)
    if i < 0 or j < 0:
        print("Something went wrong")
        return 0

    r1 = 0
    for p1 in range(i + 1):
       r1 += (rents[0][p1] * p1)

    r2 = 0
    for p2 in range(j + 1):
       r2 += (rents[1][p2] * p2)

    return (r1 + r2) * 100 - abs(move) * 20 # + i + j


def get_discount(i, j, move, util):
    return get_reward(i, j, move) + gamma * get_conditional_prob_utils(i, j, move, util)


# sum of probabilities multiplied by utilities
def get_conditional_prob_utils(i, j, move, util):
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

# def get_conditional_prob_utils(i, j, move, util):
#     # return 1
#     i, j = get_move_possible(i, j, move)
#     partial_util = np.zeros((M + 1, M + 1)).tolist()
#     partial_util_i, partial_util_j = np.zeros((M + 1)).tolist(), np.zeros((M + 1)).tolist()
#     # normalization, count all for now
#     local_sum_i, local_sum_j = 0.0, 0.0s
#     for i_poss, row in enumerate(util):
#         for j_poss, item in enumerate(row):
#             partial_sum_i, partial_sum_j = 0.0, 0.0
#             i_diff = i_poss - i     # jeśli dodatnie to wiecej zwrotow jesli ujemne to wypozyczen
#             if i_diff >= 0:     # jesli wiecej zwrotow
#                 for ret in range(i_diff, M + 1):
#                     partial_sum_i += (returns[0][ret] * rents[0][ret - i_diff])
#             else:    # wiecej wypozyczen
#                 for rent in range(abs(i_diff), M + 1):
#                     partial_sum_i += (returns[0][rent + i_diff] * rents[0][rent])
#             partial_util_i[i_poss][j_poss] = partial_sum_i
#             local_sum_j += partial_sum_i
#
#     prob = 0.0
#     for i_poss, row in enumerate(util):
#         for j_poss, item in enumerate(row):
#             prob += (get_rentals_prob(0, i_poss - i) * get_rentals_prob(1, j_poss - j)) / local_sum * util[i_poss][
#                 j_poss]
#     return prob


def get_bellman(i, j, util):
    moves_scores = []
    for move in range(-5, 6, 1):
        # check whether move is possible
        if i - move >= 0 and j + move >= 0:
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


def value_iteration():
    utilities = np.zeros((M + 1, M + 1)).tolist()
    utilities_next = np.zeros((M + 1, M + 1)).tolist()
    for r in range(20):
        utilities = utilities_next
        utilities_next = np.zeros((M + 1, M + 1)).tolist()
        print(r)
        for i, row in enumerate(utilities):
            for j, item in enumerate(row):
                utilities_next[i][j] = get_bellman(i, j, utilities)[1]
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
        if i - move >= 0 and j + move >= 0:
            moves_scores.append((move, get_conditional_prob_utils(i, j, move, utils)))
    return max(moves_scores, key=lambda p: p[1])[0]


# def assess_move(i, j, move, utils):
#     i, j = get_move_possible(i, j, move)
#     # normalization, count all for now
#     local_sum = 0.0
#     for i_poss, row in enumerate(utils):
#         for j_poss, item in enumerate(row):
#             local_sum += get_rentals_prob(0, i_poss - i) * get_rentals_prob(1, j_poss - j)
#
#     prob = 0.0
#     for i_poss, row in enumerate(utilities):
#         for j_poss, item in enumerate(row):
#             prob += (get_rentals_prob(0, i_poss - i) * get_rentals_prob(1, j_poss - j)) / local_sum * utils[i_poss][j_poss]
#     return prob


if __name__ == "__main__":
    utils = value_iteration()

    for i in range(M + 1):
        for j in range(M + 1):
            print('{:06.2f}'.format(utils[i][j]), end=' ')
        print()

    policy = get_policy(utils)

    for i in range(M + 1):
        for j in range(M + 1):
            print("{:2d} ".format(policy[i][j]), end='')
        print()


    # policy = random.randint(-5, 5, (M+1, M+1))

    # for i in range(M + 1):
    #     for j in range(M + 1):
    #         print("{:2d} ".format(policy[i][j]), end='')
    #     print()
