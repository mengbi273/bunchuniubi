# If you need to import additional packages or classes, please import here.
import numpy as np
import sys
import random

def func():
    # Driver: do not modify
    np.random.seed(42)
    sequence_length = 5
    x_dim = 7
    hidden_size = 5

    lstm_param = LstmParam(hidden_size, x_dim)
    lstm_net = LstmNetwork(lstm_param)

    # Two random sequences of shape (5,7)
    inputs = [
        np.random.rand(sequence_length, x_dim),
        np.random.rand(sequence_length, x_dim)
    ]

    for sample in inputs:
        for xt in sample:
            lstm_net.x_list_add(xt)
        for node in lstm_net.lstm_node_list:
            print(round(float(node.state.h[0]), 6))
        print("\n")
        lstm_net.x_list_clear()

if __name__ == "__main__":
    func()

