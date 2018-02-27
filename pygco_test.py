from pygco import pygco
import numpy as np

def test_float():
    unary_cost = np.array([[0.0, 1.0],
                           [4.0, 1.0],
                           [5.0, 1.0]])
    edges = np.array([[0, 1],
                      [1, 2],
                      [0, 2]]).astype(np.int32)
    pairwise_cost = np.array([[0.0, 1.0],
                              [1.0, 0.0]])
    edge_weights = np.array([0.0, 1.0, 1.0])

    n_sites = 3
    n_labels = 2
    n_edges = 3

    print pygco.cut_general_graph(edges, edge_weights, unary_cost, pairwise_cost, n_iter=1, algorithm="expansion")

test_float()

