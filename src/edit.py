#!/usr/bin/python

from __future__ import print_function

import sys, os, editdistance

if len(sys.argv) != 3:
    sys.exit('USAGE: edit.py PREDICTED_PATH TRUTH_PATH')

distances = []
accuracies = []

for file_name in os.listdir(sys.argv[1]):
    with open(os.path.join(sys.argv[1], file_name), encoding='utf8') as f:
        predicted = ''.join(f.read().split())
    with open(os.path.join(sys.argv[2], file_name), encoding='utf8') as f:
        truth = ''.join(f.read().split())
    distance = editdistance.eval(predicted, truth)
    distances.append(distance)
    accuracies.append(max(0, 1 - distance / len(truth)))
    print(f'{file_name}: {distance}')

print(f'Total distance = {sum(distances)}')
print('Average Accuracy = %.2f%%' % (sum(accuracies) / len(accuracies) * 100))
