#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import argparse


class LogParser:
    def __init__(self, log_path):
        self.log_path = log_path
        self.result = self.parse_log(log_path)

    def parse_log(self, log_path):
        result_df = pd.DataFrame()
        exp_name = None
        n_total = 0
        n_correct = 0
        with open(log_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('exp_name'):
                    exp_name = line.split()[1]
                    print(result_df)
                    n_total = 0
                    n_correct = 0
                tokens = line.split('; ')
                if len(tokens) < 6:  # not a result line
                    continue
                label = int(tokens[5].split()[1])
                preds = tokens[4][6:]
                n_majority_correct = int(tokens[1].split()[1])
                if preds == '-':
                    correct = 1
                    verified = 1
                else:
                    preds = [int(t) for t in preds[1:-1].split(', ')]
                    correct = 1 if preds[0] == label else 0
                    verified = 1 if len(set(preds)) == 1 else 0
                path = tokens[-1][5:]
                n_total += 1
                n_correct += correct
                print(f'n_total: {n_total} n_correct: {n_correct} n_majority_correct: {n_majority_correct}')
                record = {
                    'exp_name': exp_name,
                    'path': path,
                    'label': label,
                    'correct': correct,
                    'verified': verified
                }
                result_df.append(record, ignore_index=True)
                # print(f'exp_name: {exp_name}, label: {label}, path: {path}')
            return result_df

    def print(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser("Vision Model Train/Fine-tune")

    # basic config
    parser.add_argument("--log-file", type=str, help="path to the log file")
    config = parser.parse_args()

    # models config
    return config


def main():
    config = parse_args()
    if config.log_file is None:
        return
    log = LogParser(config.log_file)
    print(log.result)


if __name__ == '__main__':
    main()

