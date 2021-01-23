# SpellChecker
Levenshtein Edit Distance Benchmarking

## Benchmarks

All benchmarks performed using Threadripper 3970x @ 3.70GHz, no boost clock

| Algorithm | Single Threaded Time (s) | Multi-Threaded Time (s) |
| :-------: | :----------------------: | :---------------------: |
| levenshtein_edit_distance0 |  13.68 |  24.92 |
| levenshtein_edit_distance1 |  11.89 |  24.81 |
| levenshtein_edit_distance2 |  60.22 |  27.14 |
| levenshtein_edit_distance3 |   3.30 |  19.02 |
| levenshtein_edit_distance4 |   4.68 |  19.01 |
| levenshtein_edit_distance5 |   3.44 |  19.32 |
| levenshtein_edit_distance6 |   4.82 |  18.59 |
| levenshtein_edit_distance7 |   7.61 |  19.29 |
| levenshtein_edit_distance8 |   3.46 |  19.55 |
| levenshtein_edit_distance9 |   3.49 |  19.30 |


## levenshtein_edit_distance0
This is my first implementation of the levenshtein edit distance without any optimizations.


## levenshtein_edit_distance1
This is my second implementation of the levenshtein edit distance with branch optimizations.


## levenshtein_edit_distance2
This is my third implementation of the levenshtein edit distance with branchless optimizations.


## levenshtein_edit_distance3
This is my fourth implementation of the levenshtein edit distance with maximum optimizations.


## levenshtein_edit_distance4
This is python implementation iterative 1: https://rosettacode.org/wiki/Levenshtein_distance#Iterative_1


## levenshtein_edit_distance5
This is python implementation iterative 2: https://rosettacode.org/wiki/Levenshtein_distance#Iterative_2


## levenshtein_edit_distance6
This is python implementation iterative 3: https://rosettacode.org/wiki/Levenshtein_distance#Iterative_3


## levenshtein_edit_distance7
This is the functional implementation utilizing memoized recusion: https://rosettacode.org/wiki/Levenshtein_distance#Memoized_recursion

## levenshtein_edit_distance8
This is my fifth implementation of the levenshtein edit distance with maximum optimizations and using the array module.

## levenshtein_edit_distance9
This is my sixth implementation of the levenshtein edit distance with maximum optimizations and using the array module wihtout append.