# SpellChecker
Levenshtein Edit Distance Benchmarking

## Benchmarks

All benchmarks performed using Threadripper 3970x @ 3.70GHz, no boost clock

| Algorithm | Time (s) |
| --------- | ---- |
| levenshtein_edit_distance0 | 16.46 |
| levenshtein_edit_distance1 | 14.53 |
| levenshtein_edit_distance2 | 73.34 |
| levenshtein_edit_distance3 | 4.09 |
| levenshtein_edit_distance4 | 5.92 |
| levenshtein_edit_distance5 | 4.34 |
| levenshtein_edit_distance6 | 6.23 |
| levenshtein_edit_distance7 | 9.33 |
| levenshtein_edit_distance8 | 4.93 |
| levenshtein_edit_distance9 | 4.49 |


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