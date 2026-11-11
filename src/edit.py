#!/usr/bin/python
"""Compare predicted texts with the ground truth using edit distance.

Usage: python edit.py PREDICTED_DIR TRUTH_DIR

Files are matched by name; whitespace is ignored when comparing.
"""

import os
import sys

import editdistance


def read_text(path):
    with open(path, encoding='utf8') as f:
        return ''.join(f.read().split())


def accuracy(distance, truth_length):
    if truth_length == 0:
        # Nothing to find: perfect only if nothing was predicted.
        return 1.0 if distance == 0 else 0.0
    return max(0, 1 - distance / truth_length)


def evaluate(predicted_dir, truth_dir):
    """Return (list of (file name, distance), average accuracy in [0, 1])."""

    results = []
    accuracies = []

    for file_name in sorted(os.listdir(predicted_dir)):
        if not file_name.endswith('.txt'):
            continue

        truth_path = os.path.join(truth_dir, file_name)
        if not os.path.exists(truth_path):
            print(f'{file_name}: no truth file, skipped', file=sys.stderr)
            continue

        predicted = read_text(os.path.join(predicted_dir, file_name))
        truth = read_text(truth_path)

        distance = editdistance.eval(predicted, truth)
        results.append((file_name, distance))
        accuracies.append(accuracy(distance, len(truth)))

    average = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return results, average


def main(argv):
    if len(argv) != 3:
        sys.exit('USAGE: edit.py PREDICTED_PATH TRUTH_PATH')

    results, average = evaluate(argv[1], argv[2])
    if not results:
        sys.exit('No predicted files with a matching truth file were found.')

    for file_name, distance in results:
        print(f'{file_name}: {distance}')

    print(f'Total distance = {sum(distance for _, distance in results)}')
    print('Average Accuracy = %.2f%%' % (average * 100))


if __name__ == '__main__':
    main(sys.argv)
