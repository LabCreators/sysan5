import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import deepcopy
from consts import fuzzy_sets_consts


alpha_root2_or = np.array([[0.5, None, 0.2, None, 0.8, 0.8, 0.2],
                       [None, 0.55, 0.2, None, 0.2, 0.1, 0.9],
                       [0.1, 0.23, 0.3, None, 0.4, None, None],
                       [0.3, 0.25, 0.4, 0.55, 0.5, 0.6, None]])


Ip_root2_or = np.array([[0.3, None, 0.1, None, 0.54, 0.23, 0.12],
                       [None, 0.9, 0.8, None, 0.2, 0.34, 0.76],
                       [0.45, 0.34, 0.3, None, 0.6, None, None],
                       [0.5, 0.6, 0.3, 0.35, 0.4, 0.7, None]])


Id_root2_or = np.array([[0.9, None, 0.3, None, 0.1, 0.2, 0.3],
                       [None, 0.34, 0.6, None, 0.5, 0.2, 0.25],
                       [0.13, 0.3, 0.37, None, 0.46, None, None],
                       [0.33, 0.5, 0.47, 0.555, 0.65, 0.46, None]])


It_root2_or = np.array([[0.5, None, 0.2, None, 0.81, 0.81, 0.22],
                       [None, 0.55, 0.2, None, 0.22, 0.1, 0.9],
                       [0.105, 0.23, 0.3, None, 0.42, None, None],
                       [0.35, 0.15, 0.8, 0.85, 0.7, 0.6, None]])




alpha_root_or = np.array([[0.2, None, 0.3, 0.45, 0.7, 0.3, 0.5],
                       [None, 0.4, 0.3, None, 0.6, 0.5, None],
                       [None, 0.35, 0.4, None, 0.4, 0.7, None],
                       [0.3, 0.25, 0.4, 0.55, 0.5, None, None]])


Ip_root_or = np.array([[0.4, None, 0.5, 0.5, 0.8, 0.4, 0.6],
                   [None, 0.6, 0.5, None, 0.3, 0.6, None],
                   [None, 0.4, 0.4, None, 0.3, 0.75, None],
                   [0.5, 0.4, 0.4, 0.65, 0.3, None, None]])


Id_root_or = np.array([[0.6, None, 0.7, 0.4, 0.7, 0.45, 0.6],
                   [None, 0.5, 0.75, None, 0.4, 0.3, None],
                   [None, 0.3, 0.55, None, 0.45, 0.8, None],
                   [0.7, 0.3, 0.65, 0.4, 0.65, None, None]])


It_root_or = np.array([[0.4, None, 0.5, 0.5, 0.8, 0.5, 0.8],
                    [None, 0.5, 0.6, None, 0.6, 0.4, None],
                    [None, 0.6, 0.7, None, 0.4, 0.3, None],
                    [0.5, 0.7, 0.8, 0.6, 0.2, None, None]])


def get_Ip(t, alpha, Ip_prev):
    temp = Ip_prev * (1 + 0.5 * (1 + alpha) * t)
    if 0 < temp < 1:
        return temp
    return 1


def get_Id(t, gamma, Id_prev):
    temp = Id_prev * (1 + (1 + gamma) * t)
    if 0 < temp < 1:
        return temp
    return 1


def get_It(t, gamma, It_prev, T1, T2):
    temp = It_prev * (1 - gamma * t)
    if 0 < temp <= 1 and T1 <= t <= T2:
        return temp
    return 0


def get_alpha(alpha_prev, Ip_prev):
    if 0 < alpha_prev < 1:
        return alpha_prev * Ip_prev / (1 + alpha_prev)
    return 0


def get_beta(alpha_prev, Id_prev, It_prev):
    if 0 < alpha_prev < 1:
        return 0.01 * It_prev / (1 + alpha_prev * Id_prev)
    return 0


def get_gamma(alpha_prev, beta, Id_prev):
    if 0 < alpha_prev < 1:
        return (1 + alpha_prev * Id_prev) * (1 + alpha_prev * Id_prev) / (100 * beta + 5 * (1 + alpha_prev * alpha_prev * alpha_prev))
    return 0


def get_entropy(t, alpha, beta, gamma, Ip, Id, It):
    return 1 - math.log2(1 + It * Ip * Id * (1 + alpha * t) * (1 + gamma * t) * (1 + beta * t) * (7 + gamma))


def get_one_history(data=None, name=None, index_i=0, index_j=0):
    li = []
    for index_time in range(len(data)):
        li.append((data.loc[index_time]).I[name][index_i, index_j])
    return li


def graph_plot(t, dataset, I_name, index_i, index_j, x_label='time, s'):
    fix, ax = plt.subplots()
    ax.plot(t, get_one_history(dataset, I_name, index_i, index_j))
    ax.set(xlabel=x_label, ylabel=I_name, title='Physics of {}[{}, {}]'.format(I_name, index_i, index_j))
    ax.grid()
    fix.savefig("photos/Physics of_{}[{}, {}].png".format(I_name, index_i, index_j))
    return fix


def init_decision_time(example):
    temp = deepcopy(example)
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            if temp[i, j] is not None:
                temp[i, j] = [0, 'infinity']
    return temp

def check_infinity(value):
    try:
        int(value)
        return True
    except:
        return False

def find_intervals_per_situation(decision_time_matrix):
    return [(min([el[0] for el in x if el]), min([el[1] for el in x if el])) for x in decision_time_matrix]


def predict(intervals, eta):
    df_result = pd.DataFrame(pd.Series(intervals, index=['S{}'.format(str(i)) for i in range(1, len(intervals)+1)]),
                             columns=['intervals'])

    min_bound, max_bound = fuzzy_sets_consts.get(eta)
    df_result['class'] = df_result.intervals.apply(lambda x: 'Calm' if x[1] > max_bound else 'Panic' if x[1] < min_bound
                                                   else 'Dunno')

    return df_result

def getDecisionTime(eta=0.6, own=False):
    # initialize necessary tensors
    if own:
        Ip_root = deepcopy(Ip_root2_or)
        Id_root = deepcopy(Id_root2_or)
        It_root = deepcopy(It_root2_or)
        alpha_root = deepcopy(alpha_root2_or)
    else:
        Ip_root = deepcopy(Ip_root_or)
        Id_root = deepcopy(Id_root_or)
        It_root = deepcopy(It_root_or)
        alpha_root = deepcopy(alpha_root_or)
    df_history = pd.DataFrame(columns=['t', 'I'])
    decision_time = init_decision_time(Ip_root)

    t = [i / 100 for i in range(0, 200)]
   # t = [i for i in range(0, 100)]

    for i in t:
        df_history.loc[len(df_history)] = [i,
                                           {'Ip': deepcopy(Ip_root), 'Id': deepcopy(Id_root), 'It': deepcopy(It_root)}]

        for j in range(len(alpha_root)):
            for k in range(len(alpha_root[0])):

                if alpha_root[j, k] is not None:

                    alpha = get_alpha(alpha_root[j, k], Ip_root[j, k])
                    beta = get_beta(alpha_root[j, k], Id_root[j, k], It_root[j, k])
                    gamma = get_gamma(alpha_root[j, k], beta, Id_root[j, k])
                    Ip = get_Ip(i, alpha, Ip_root[j, k])
                    Id = get_Id(i, gamma, Id_root[j, k])
                    T2 = decision_time[j, k][1] if check_infinity(decision_time[j, k][1]) else max(t)
                    It = get_It(i, gamma, It_root[j, k], T1=min(t), T2=T2)

                    entropy = get_entropy(i, alpha, beta, gamma, Ip, Id, It)

                    if entropy > eta and check_infinity(decision_time[j, k][1]) == False:
                        decision_time[j, k] = [0, i]

                    # rewrite previous
                    alpha_root[j, k] = alpha
                    Ip_root[j, k] = Ip
                    Id_root[j, k] = Id
                    It_root[j, k] = It

    intervals_tuples_list = find_intervals_per_situation(decision_time)
    classification_result = predict(intervals_tuples_list, eta)

    result = []
    for index_row, row in enumerate(decision_time):
        result.append([])
        for index_column, column in enumerate(row):
            result[index_row].append(str(column))
    return result, df_history, t, classification_result['class'].iloc


if __name__ == "__main__":
    result, df_history, t, classification_result = getDecisionTime(eta=.2)
    print(classification_result)
