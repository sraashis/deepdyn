import math as mth
import networkx as nx
import numpy as np


__all__ = [
    'create_lattice_graph(image_arr_2d)',
    'assign_cost(graph=nx.Graph(), images=[()], alpha=1, override=False, log=False)',
    'assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0)))'
]


def create_lattice_graph(image_arr_2d):
    graph = nx.grid_graph([image_arr_2d.shape[0], image_arr_2d.shape[1]])
    n_pos = dict(zip(graph.nodes(), graph.nodes()))
    # for i, j in graph.nodes():
    #     n0 = (i, j)
    #     n1 = (i - 1, j + 1)
    #     n2 = (i + 1, j - 1)
    #     n3 = (i - 1, j - 1)
    #     n4 = (i + 1, j + 1)
    #     if n1 in graph.nodes():
    #         graph.add_edge(n0, n1)
    #     if n2 in graph.nodes():
    #         graph.add_edge(n0, n2)
    #     if n3 in graph.nodes():
    #         graph.add_edge(n0, n3)
    #     if n4 in graph.nodes():
    #         graph.add_edge(n0, n4)
    return graph, n_pos


def assign_cost(graph=nx.Graph(), images=[()], alpha=1, override=False, log=False):
    i = 0
    for n1 in graph.nodes():
        for n2 in nx.neighbors(graph, n1):
            if graph[n1][n2] == {} or override:
                cost = 0.0
                ix = 1
                for weight, arr in images:
                    i_diff = abs(float(arr[n1[0], n1[1]]) - float(arr[n2[0], n2[1]]))
                    cost += weight * mth.pow(mth.e, alpha * (i_diff / 255))
                    graph[n1][n2]['i_diff_' + str(ix)] = i_diff
                    ix += 1
                graph[n1][n2]['cost'] = cost
        if log:
            print('\r' + str(i) + ': ' + str(n1), end='')
            i += 1


def assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0))):
    for i in range(metrics.shape[0]):
        print('\r' + str(i), end='')
        for j in range(metrics.shape[1]):
            graph[(i, j)]["skeleton"] = metrics[i, j]
