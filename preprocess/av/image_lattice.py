import itertools as itr
import math as mth

import networkx as nx


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


def assign_cost(graph=nx.Graph(), images={}, alpha=1):
    for n1, n2 in itr.combinations(graph.nodes(), 2):
        if n1 in graph.neighbors(n2):
            cost = 0.0
            for weight, arr in images.items():
                i_diff = abs(int(arr[n1[0], n1[1]]) - int(arr[n2[0], n2[1]]))
                cost += float(weight) / float(1 + mth.pow(mth.e, -(float(alpha) * i_diff)))
            graph[n1][n2]['cost'] = cost

