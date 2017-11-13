import math as mth
from multiprocessing import Process

import networkx as nx
import numpy as np

from commons.LOGGER import Logger
from commons.timer import check_time

__all__ = [
    'create_lattice_graph(image_arr_2d)',
    'assign_cost(graph=nx.Graph(), images=[()], alpha=1, override=False, log=False)',
    'assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0)))'
]


def get_sub_lattice(i, j, x_block_size, y_block_size):
    for p in range(i, i + x_block_size, 1):
        for q in range(j, j + y_block_size, 1):
            yield (p, q)


def chunk_lattice(image_arr_2d, full_lattice, grid_size=(0, 0)):
    sub_graphs_nodes = []
    x_limit, y_limit = image_arr_2d.shape
    x_block_size = int(x_limit / grid_size[0])
    y_block_size = int(y_limit / grid_size[1])

    remain_x = x_limit % x_block_size
    x_end = x_limit - remain_x
    remain_y = y_limit % y_block_size
    y_end = y_limit - remain_y

    for i in range(0, x_end, x_block_size):
        for j in range(0, y_end, y_block_size):
            x_size = x_block_size
            y_size = y_block_size
            if i + x_block_size == x_end:
                x_size = x_block_size + remain_x
            if j + y_block_size == y_end:
                y_size = y_block_size + remain_y
            Logger.log(str(i) + ',' + str(j))
            sub_graphs_nodes.append(nx.subgraph(full_lattice, get_sub_lattice(i, j, x_size, y_size)))
    return sub_graphs_nodes


def generate_lattice_graph(image_arr_2d):
    graph = nx.grid_graph([image_arr_2d.shape[0], image_arr_2d.shape[1]])
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
    return graph


@check_time
def assign_cost(graph=nx.Graph(), images=[()], alpha=10, override=False, log=False):
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


def assign_cost_parallel(lattices, images=[()], alpha=10, override=False):
    all_p = []
    for a_lattice in lattices:
        p = Process(target=assign_cost(a_lattice, images, alpha, override, True))
        all_p.append(p)
    for p in all_p:
        p.run()
    for p in all_p:
        p.join()


def assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0))):
    for i, j in graph.nodes():
        graph[(i, j)]['skeleton'] = metrics[i, j]
