#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import poisson


if __name__ == "__main__":
    gamma = 0.9
    move_cost = 20
    car_reward = 100
    rents = []
    returns = []

    rents.append([poisson.pmf(x, mu = 3) for x in range(21)])
    rents.append([poisson.pmf(x, mu = 4) for x in range(21)])
    returns.append([poisson.pmf(x, mu = 3) for x in range(21)])
    returns.append([poisson.pmf(x, mu = 2) for x in range(21)])

    utilities = np.zeros(21,21)



    # M = 20
    # policy = random.randint(-5, 5, (M+1, M+1))

    # for i in range(M + 1):
    #     for j in range(M + 1):
    #         print("{:2d} ".format(policy[i][j]), end='')
    #     print()