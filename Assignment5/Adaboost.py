# !bin/bash
#  -*- coding:utf-8 -*-
import numpy as np
import math
from NaiveBayes import data_pre, ten_fold_cross_validation_split, test_validate


def init_weights(N):
    return np.ones(N) * (float)(1.0 / N)


def update_weights():
    pass

if __name__ == "__main__":
    train_y, train_x, types_row = data_pre("./", "german-assignment5.txt")
    weights = init_weights(train_x.shape[0])


