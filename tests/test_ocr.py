"""Smoke and regression tests. Run from the repository root: pytest"""

import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, SRC)

import edit  # noqa: E402
import OCR  # noqa: E402
import utilities  # noqa: E402
from character_segmentation import vertical_transitions  # noqa: E402
from segmentation import projection_segmentation  # noqa: E402
from train import prepare_char  # noqa: E402

TRUTH_DIR = os.path.join(SRC, 'truth')
# Measured accuracy on the bundled samples is 95.7%; leave room for
# floating-point / library-version noise but catch real regressions.
MIN_ACCURACY = 0.93


def test_last_segment_touching_the_edge_is_kept():
    img = np.zeros((20, 60), np.uint8)
    img[:, 5:15] = 255
    img[:, 40:60] = 255          # touches the right edge
    assert len(projection_segmentation(img, 'vertical', cut=3)) == 2

    img = np.zeros((60, 20), np.uint8)
    img[5:15, :] = 255
    img[40:60, :] = 255          # touches the bottom edge
    assert len(projection_segmentation(img, 'horizontal', cut=3)) == 2


@pytest.mark.parametrize('crop', [
    None,
    np.zeros((10, 0), np.uint8),   # empty
    np.zeros((10, 6), np.uint8),   # no ink
])
def test_prepare_char_returns_none_for_unusable_images(crop):
    assert prepare_char(crop) is None


def test_prepare_char_resizes_to_25x25():
    crop = np.zeros((12, 8), np.uint8)
    crop[2:10, 2:6] = 255
    assert prepare_char(crop).shape == (25, 25)


@pytest.mark.parametrize('column, expected', [
    ([0, 0, 0], 0),
    ([1, 1, 1], 1),            # a run that reaches the bottom still counts once
    ([0, 1, 1, 0], 2),
    ([1, 0, 1, 0, 1], 5),
    ([0, 1, 0, 1, 1, 0, 1], 5),
])
def test_vertical_transitions(column, expected):
    img = np.array(column, np.uint8).reshape(-1, 1)
    assert vertical_transitions(img, 0) == expected
    assert vertical_transitions(img.astype(bool), 0) == expected


def test_projection_rejects_unknown_axis():
    with pytest.raises(ValueError):
        utilities.projection(np.zeros((2, 2)), 'diagonal')


def test_edit_accuracy_handles_empty_truth():
    assert edit.accuracy(0, 0) == 1.0
    assert edit.accuracy(3, 0) == 0.0
    assert edit.accuracy(5, 10) == 0.5


def test_image_roundtrip_with_non_ascii_path(tmp_path):
    path = str(tmp_path / 'ا.png')
    utilities.write_image(path, np.full((5, 5), 255, np.uint8))
    assert utilities.read_image(path, 0).shape == (5, 5)


def test_end_to_end_on_bundled_samples(tmp_path):
    exit_code = OCR.main(['--output', str(tmp_path)])
    assert exit_code == 0

    results, average = edit.evaluate(str(tmp_path / 'text'), TRUTH_DIR)
    assert {name for name, _ in results} == {'capr12.txt', 'capr14.txt'}
    assert average >= MIN_ACCURACY, f'accuracy dropped to {average:.2%}'
