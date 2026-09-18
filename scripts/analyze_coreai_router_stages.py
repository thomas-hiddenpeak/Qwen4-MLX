#!/usr/bin/env python3
"""CPU-only attribution after root executes paired router diagnostic assets."""
import argparse
import json
from pathlib import Path

import numpy as np


def difference(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    delta = a-b
    return {'exact': bool(np.array_equal(actual, expected)), 'different': int(np.count_nonzero(actual != expected)),
        'maxAbsoluteError': float(np.max(np.abs(delta))),
        'relativeL2Error': float(np.linalg.norm(delta)/np.linalg.norm(b)) if np.any(b) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic', type=Path)
    parser.add_argument('--prior', type=Path, help='Original non-diagnostic paired result directory, if available')
    args = parser.parse_args()
    expected_json = json.loads((args.diagnostic/'actual.json').read_text())['expectedOutputs']
    expected = {name: np.array(value['values'], dtype=np.int32 if name == 'ids' else np.float16).reshape(value['shape'])
                for name, value in expected_json.items()}
    report = {'status': 'CPU-analysis-of-device-files', 'variants': {}}
    for variant in ('baseline', 'candidate'):
        directory = args.diagnostic/(variant+'-output')
        actual = {name: np.fromfile(directory/(name+'.bin'), dtype=reference.dtype).reshape(reference.shape)
                  for name, reference in expected.items()}
        selected = np.take_along_axis(actual['probabilities'], actual['ids'], axis=1)
        denominator = selected[:, 0].copy()
        for index in range(1, 10):
            denominator = (denominator.astype(np.float32)+selected[:, index].astype(np.float32)).astype(np.float16)
        divided = (selected.astype(np.float32)/actual['normalizer'].astype(np.float32)).astype(np.float16)
        replayed = (selected.astype(np.float32)/denominator[:, None].astype(np.float32)).astype(np.float16)
        entry = {'deviceVersusCPU': {name: difference(actual[name], value) for name, value in expected.items()},
            'deviceDenominatorVersusSequentialHalfReplay': difference(actual['normalizer'][:, 0], denominator),
            'deviceScoresVersusFloatDivideThenHalfOfDeviceOperands': difference(actual['scores'], divided),
            'deviceScoresVersusHalfChainReplayFromDeviceProbabilities': difference(actual['scores'], replayed)}
        if args.prior:
            prior = np.fromfile(args.prior/(variant+'-output/scores.bin'), dtype=np.float16).reshape(actual['scores'].shape)
            entry['diagnosticScoresVersusPriorNonDiagnostic'] = difference(actual['scores'], prior)
        report['variants'][variant] = entry
    output = args.diagnostic/'stage-analysis.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
