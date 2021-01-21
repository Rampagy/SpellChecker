'''
Spell Checker using the Levenshtein Edit Distance via the optimized Wagner-Fischer algorithm.

References:
https://www.youtube.com/watch?v=Cu7Tl7FGigQ
http://www.gwicks.net/dictionaries.htm
'''

import csv
import time

import numpy as np
import multiprocessing as mp

from array import array
from queue import PriorityQueue
from functools import lru_cache


def levenshtein_edit_distance0(word1, word2):
    # This is my 1st implementation of the levenshtein edit distance

    levenshtein_previous_row = np.zeros(len(word1) + 1, dtype=np.int)
    levenshtein_current_row = np.zeros(len(word1) + 1, dtype=np.int)

    for word2_idx in range(len(word2) + 1):
        for word1_idx in range(levenshtein_previous_row.shape[0]):

            # Fill out first row
            if word2_idx == 0:
                levenshtein_current_row[word1_idx] = word1_idx

            # Fill out first index
            elif word1_idx == 0:
                levenshtein_current_row[word1_idx] = word2_idx

            # If the letters match use the kitty corner value
            elif word1[word1_idx - 1] == word2[word2_idx - 1]:
                levenshtein_current_row[word1_idx] = levenshtein_previous_row[word1_idx - 1]

            # If the letters don't match use the minimum of the three surrounding values plus 1
            else:
                levenshtein_current_row[word1_idx] = min( levenshtein_previous_row[word1_idx], levenshtein_previous_row[word1_idx - 1], levenshtein_current_row[word1_idx - 1] ) + 1

        # copy current row to previous row
        levenshtein_previous_row = np.copy(levenshtein_current_row)

    return levenshtein_current_row[levenshtein_current_row.shape[0] - 1]


def levenshtein_edit_distance1(word1, word2):
    # This is my 2nd implementation of the levenshtein edit distance

    levenshtein_previous_row = np.arange(len(word1) + 1, dtype=np.int)
    levenshtein_current_row = np.zeros(len(word1) + 1, dtype=np.int)

    for word2_idx, char2 in enumerate(word2):
        levenshtein_current_row[0] = word2_idx + 1
        for word1_idx, char1 in enumerate(word1):
            if char1 == char2:
                levenshtein_current_row[word1_idx+1] = levenshtein_previous_row[word1_idx]
            else:
                levenshtein_current_row[word1_idx+1] = 1 + min( levenshtein_previous_row[word1_idx + 1],
                                                              levenshtein_previous_row[word1_idx],
                                                              levenshtein_current_row[word1_idx] )

        # copy current row to previous row
        levenshtein_previous_row = np.copy(levenshtein_current_row)

    return levenshtein_current_row[levenshtein_current_row.shape[0] - 1]


def levenshtein_edit_distance2(word1, word2):
    # This is my 3rd implementation of the levenshtein edit distance
    # Branchless programming attempt

    levenshtein_previous_row = np.arange(len(word1) + 1, dtype=np.int)
    levenshtein_current_row = np.zeros(len(word1) + 1, dtype=np.int)

    for word2_idx, char2 in enumerate(word2):
        levenshtein_current_row[0] = word2_idx + 1
        for word1_idx, char1 in enumerate(word1):
            levenshtein_current_row[word1_idx+1] = (char1 == char2) * levenshtein_previous_row[word1_idx] + \
                                                   (char1 != char2) * ( 1 + min( levenshtein_previous_row[word1_idx + 1],
                                                                                levenshtein_previous_row[word1_idx],
                                                                                levenshtein_current_row[word1_idx] ) )

        # copy current row to previous row
        levenshtein_previous_row = np.copy(levenshtein_current_row)

    return levenshtein_current_row[levenshtein_current_row.shape[0] - 1]


def levenshtein_edit_distance3(word1, word2):
    # This is my 3rd implementation of the levenshtein edit distance

    levenshtein_previous_row = range(len(word1) + 1)
    for word2_idx, char2 in enumerate(word2):
        levenshtein_current_row = [word2_idx + 1]
        for word1_idx, char1 in enumerate(word1):
            if char1 == char2:
                levenshtein_current_row.append(levenshtein_previous_row[word1_idx])
            else:
                levenshtein_current_row.append(1 + min( levenshtein_previous_row[word1_idx + 1],
                                                        levenshtein_previous_row[word1_idx],
                                                        levenshtein_current_row[word1_idx] ) )

        # copy current row to previous row
        levenshtein_previous_row = levenshtein_current_row

    return levenshtein_current_row[-1]


def levenshtein_edit_distance4(str1, str2):
    # This is python implementation iterative 1
    # https://rosettacode.org/wiki/Levenshtein_distance#Iterative_1

    m = len(str1)
    n = len(str2)
    d = [[i] for i in range(1, m + 1)]   # d matrix rows
    d.insert(0, list(range(0, n + 1)))   # d matrix columns
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:   # Python (string) is 0-based
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i].insert(j, min(d[i - 1][j] + 1,
                               d[i][j - 1] + 1,
                               d[i - 1][j - 1] + substitutionCost))
    return d[-1][-1]


def levenshtein_edit_distance5(s1, s2):
    # This is python implementation iterative 2
    # https://rosettacode.org/wiki/Levenshtein_distance#Iterative_2

    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


def levenshtein_edit_distance6(a, b, mx=-1):
    # This is python implementation iterative 3
    # https://rosettacode.org/wiki/Levenshtein_distance#Iterative_3
    def result(d): return d if mx < 0 else False if d > mx else True

    if a == b: return result(0)
    la, lb = len(a), len(b)
    if mx >= 0 and abs(la - lb) > mx: return result(mx+1)
    if la == 0: return result(lb)
    if lb == 0: return result(la)
    if lb > la: a, b, la, lb = b, a, lb, la

    cost = array('i', range(lb + 1))
    for i in range(1, la + 1):
        cost[0] = i; ls = i-1; mn = ls
        for j in range(1, lb + 1):
            ls, act = cost[j], ls + int(a[i-1] != b[j-1])
            cost[j] = min(ls+1, cost[j-1]+1, act)
            if (ls < mn): mn = ls
        if mx >= 0 and mn > mx: return result(mx+1)
    if mx >= 0 and cost[lb] > mx: return result(mx+1)
    return result(cost[lb])


@lru_cache(maxsize=4095)
def levenshtein_edit_distance7(s, t):
    # This is the functional implementation utilizing memoized recusion
    # https://rosettacode.org/wiki/Levenshtein_distance#Memoized_recursion

    if not s: return len(t)
    if not t: return len(s)
    if s[0] == t[0]: return levenshtein_edit_distance7(s[1:], t[1:])
    l1 = levenshtein_edit_distance7(s, t[1:])
    l2 = levenshtein_edit_distance7(s[1:], t)
    l3 = levenshtein_edit_distance7(s[1:], t[1:])
    return 1 + min(l1, l2, l3)


def levenshtein_edit_distance8(word1, word2):
    # This is my 8th implementation of the levenshtein edit distance
    # cost = array('i', range(lb + 1))

    levenshtein_previous_row = array('I', range(len(word1) + 1))
    for word2_idx, char2 in enumerate(word2):
        levenshtein_current_row = array('I', [word2_idx + 1])
        for word1_idx, char1 in enumerate(word1):
            if char1 == char2:
                levenshtein_current_row.append(levenshtein_previous_row[word1_idx])
            else:
                levenshtein_current_row.append(1 + min( levenshtein_previous_row[word1_idx + 1],
                                                        levenshtein_previous_row[word1_idx],
                                                        levenshtein_current_row[word1_idx] ) )

        # copy current row to previous row
        levenshtein_previous_row = levenshtein_current_row

    return levenshtein_current_row[-1]


def levenshtein_edit_distance9(word1, word2):
    # This is my 6th implementation of the levenshtein edit distance

    levenshtein_previous_row = array('I', range(len(word1) + 1))

    for word2_idx, char2 in enumerate(word2):
        levenshtein_current_row = array('I', [word2_idx + 1])
        for word1_idx, char1 in enumerate(word1):
            if char1 == char2:
                levenshtein_current_row.append(levenshtein_previous_row[word1_idx])
            else:
                levenshtein_current_row.append(1 + min( levenshtein_previous_row[word1_idx + 1],
                                                        levenshtein_previous_row[word1_idx],
                                                        levenshtein_current_row[word1_idx] ) )

        # copy current row to previous row
        levenshtein_previous_row = levenshtein_current_row

    return levenshtein_current_row[-1]


def levenshtein_edit_distance0_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance0(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance1_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance1(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance2_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance2(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance3_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance3(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance4_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance4(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance5_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance5(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance6_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance6(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance7_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance7(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance8_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance8(word1, word2)
    q.put((edit_dist, word2.upper()))


def levenshtein_edit_distance9_wrapper(q, word1, word2):
    edit_dist = levenshtein_edit_distance9(word1, word2)
    q.put((edit_dist, word2.upper()))


if __name__ == '__main__':
    # create list to store the english dictionary
    dictionary = []

    # load dictionary
    with open('../english3.txt', 'r') as english_dict:
        reader = csv.reader(english_dict)
        for row in reader:
            dictionary += row

    # create list of all the functions to be tested
    levenshtein_functions = [levenshtein_edit_distance0,
                             levenshtein_edit_distance1,
                             levenshtein_edit_distance2,
                             levenshtein_edit_distance3,
                             levenshtein_edit_distance4,
                             levenshtein_edit_distance5,
                             levenshtein_edit_distance6,
                             levenshtein_edit_distance7,
                             levenshtein_edit_distance8,
                             levenshtein_edit_distance9]

    wrapper_functions = [levenshtein_edit_distance0_wrapper,
                         levenshtein_edit_distance1_wrapper,
                         levenshtein_edit_distance2_wrapper,
                         levenshtein_edit_distance3_wrapper,
                         levenshtein_edit_distance4_wrapper,
                         levenshtein_edit_distance5_wrapper,
                         levenshtein_edit_distance6_wrapper,
                         levenshtein_edit_distance7_wrapper,
                         levenshtein_edit_distance8_wrapper,
                         levenshtein_edit_distance9_wrapper]

    # loop through each function for benchmarking
    for func, wrapper in zip(levenshtein_functions, wrapper_functions):

        # Test single-threaded
        levenshtein_edit_distances = PriorityQueue()
        start_time = time.time()

        for word in dictionary:
            edit_dist = func('CAVVAGES', word.upper())
            levenshtein_edit_distances.put((edit_dist, word.upper()))
        single_thread_time = time.time() - start_time


        # Test multi-threaded
        m = mp.Manager()
        levenshtein_edit_distances = m.Queue()
        pool_tuple = [(levenshtein_edit_distances, 'CAVVAGES', word) for word in dictionary]

        start_time = time.time()
        with mp.Pool(mp.cpu_count()) as p:
            p.starmap(wrapper, pool_tuple)
        multi_thread_time = time.time() - start_time

        print('| {:25s} | {:6.2f} | {:6.2f} |'.format(func.__name__, single_thread_time, multi_thread_time))