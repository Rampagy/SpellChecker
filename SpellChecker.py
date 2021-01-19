'''
Spell Checker using the Levenshtein Edit Distance via the optimized Wagner-Fischer algorithm.

References:
https://www.youtube.com/watch?v=Cu7Tl7FGigQ
http://www.gwicks.net/dictionaries.htm
'''

import csv
import numpy as np
from queue import PriorityQueue


def levenshtein_edit_distance(word1, word2):
    levenshtein_previous_row = np.zeros(len(word1) + 1, dtype=np.ushort)
    levenshtein_current_row = np.zeros(len(word1) + 1, dtype=np.ushort)

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


if __name__ == '__main__':
    # create list to store the english dictionary
    dictionary = []

    # create list to store the result of the levenshtein edit distance
    levenshtein_edit_distances = PriorityQueue()

    # load dictionary
    with open('english3.txt', 'r') as english_dict:
        reader = csv.reader(english_dict)
        for row in reader:
            dictionary += row

    # Find the levenshtein edit distance for each word in the dictionary
    for word in dictionary:
        edit_dist = levenshtein_edit_distance('CAVVAGES', word.upper())
        levenshtein_edit_distances.put((edit_dist, word.upper()))

    for _ in range(20):
        print(levenshtein_edit_distances.get())
