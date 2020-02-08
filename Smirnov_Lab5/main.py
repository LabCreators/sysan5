import pandas as pd
import numpy as np
import math
from copy import deepcopy


alpha_root = np.array([[0.6, 0.5, 0.4, 0.55, None, 0.755, 0.45],
                       [None, None, 0.7, 0.35, None, None, 0.7],
                       [0.65, None, 0.8, 0.65, 0.7, 0.65, 0.7],
                       [None, None, 0.4, 0.55, 0.45, 0.85, None]])


Ip_root = np.array([[0.65, 0.55, 0.8, 0.45, None, 0.7, 0.75],
                   [None, None, 0.5, 0.8, None, None, 0.45],
                   [0.45, None, 0.6, 0.5, 0.6, 0.45, 0.45],
                   [None, None, 0.7, 0.7, 0.4, 0.35, None]])


Id_root = np.array([[0.3, 0.54, 0.5, 0.4, None, 0.5, 0.4],
                   [None, None, 0.65, 0.25, None, None, 0.5],
                   [0.35, None, 0.35, 0.4, 0.2, 0.45, 0.3],
                   [None, None, 0.65, 0.5, 0.65, 0.3, None]])


It_root = np.array([[0.7, 0.8, 0.4, 0.6, None, 0.85, 0.5],
                    [None, None, 0.5, 0.4, None, None, 0.4],
                    [0.25, None, 0.76, 0.45, 0.3, 0.5, 0.6],
                    [None, None, 0.5, 0.5, 0.6, 0.3, None]])


def get_Ip(t, gamma, alpha, Ip_prev):
    temp = 1 + 0.25 * Ip_prev * (1 + 0.5 * alpha) * (1 + gamma) * t * (1 + 0.5 * alpha) * (1 + gamma) * t
    if 0 < temp < 1:
        return temp
    return 1


def get_Id(t, alpha_prev, Id_prev, Ip_prev):
    temp = 1 + 0.25 * (1 + Id_prev) * (1 + alpha_prev * t) * (1 + alpha_prev * t) / ((1 + 10 * Ip_prev) * (1 + 10 * Ip_prev))
    if 0 < temp < 1:
        return temp
    return 1

def get_It(t, beta, It_prev, T1, T2):
    temp = 1 - 0.25 * It_prev * (1 + (1 + beta * t) * (1 + beta * t))
    if 0 < temp < 1 and T1 <= t <= T2:
        return temp
    return 0


def get_alpha(alpha_prev, It_prev, Ip_prev):
    if 0 < alpha_prev < 1:
        return 1 + 0.5 * (It_prev + Ip_prev) * alpha_prev
    return 0


def get_beta(alpha_prev, It_prev):
    if 0 < alpha_prev < 1:
        return 1 + 0.5 * (1 + 0.01 * alpha_prev) * (1 + 0.01 * alpha_prev) / (It_prev * It_prev * (1 + alpha_prev))
    return 0


def get_gamma(alpha_prev, Id_prev, It_prev):
    if 0 < alpha_prev < 1:
        return (1 + alpha_prev * alpha_prev * Id_prev) / (1 + It_prev)
    return 0


def get_entropy(t, alpha, beta, gamma, Ip, Id, It):
    return 1 - math.log2(1 + alpha * It * Ip * Id * (1 + alpha * t) * (1 + gamma * t) * (1 - beta * t * t))


if __name__ == "__main__":
    eta = 0.6  # user should write this value on UI

    df_history = pd.DataFrame(columns=['t', 'I'])
    t = [i for i in range(0, 10)]

    for i in t:
        df_history.loc[len(df_history)] = [i, {'Ip': deepcopy(Ip_root), 'Id': deepcopy(Id_root), 'It': deepcopy(It_root)}]

        for j in range(len(alpha_root)):
            for k in range(len(alpha_root[0])):

                if alpha_root[j, k] is not None:

                    alpha = get_alpha(alpha_root[j, k], It_root[j, k], Ip_root[j, k])
                    beta = get_beta(alpha_root[j, k], It_root[j, k])
                    gamma = get_gamma(alpha_root[j, k], Id_root[j, k], It_root[j, k])
                    Ip = get_Ip(i, gamma, alpha, Ip_root[j, k])
                    Id = get_Id(i, alpha_root[j, k], Id_root[j, k], Ip_root[j, k])
                    It = get_It(i, beta, It_root[j, k], T1=0, T2=1000)

                    entropy = get_entropy(i, alpha, beta, gamma, Ip, Id, It)
                    print("t={}, entropy={}".format(i, entropy))
                    if 0 <= entropy <= eta:
                        hahah = 0

                    # rewrite previous alpha
                    alpha_root[j, k] = alpha
                    Ip_root[j, k] = Ip
                    Id_root[j, k] = Id
                    It_root[j, k] = It

    #print((df_history.loc[5]).I['It'])
