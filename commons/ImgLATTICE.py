import logging as logger
import math as mth

import networkx as nx
import numpy as np

import commons.constants as const


class Lattice:
    def __init__(self, image_arr_2d):
        logger.basicConfig(level=logger.INFO)
        self.x_size, self.y_size = image_arr_2d.shape
        self.graph = None
        self.accumulator = np.zeros([self.x_size, self.y_size], dtype=np.uint8)
        self.total_weight = 0.0

    @staticmethod
    def _connect_8(graph):
        for i, j in graph:
            n0 = (i, j)
            n1 = (i - 1, j + 1)
            n2 = (i + 1, j - 1)
            n3 = (i - 1, j - 1)
            n4 = (i + 1, j + 1)
            if n1 in graph.nodes():
                graph.add_edge(n0, n1)
            if n2 in graph.nodes():
                graph.add_edge(n0, n2)
            if n3 in graph.nodes():
                graph.add_edge(n0, n3)
            if n4 in graph.nodes():
                graph.add_edge(n0, n4)

    def generate_lattice_graph(self, eight_connected=const.IMG_LATTICE_EIGHT_CONNECTED):
        self.graph = nx.grid_2d_graph(self.x_size, self.y_size)
        if eight_connected:
            Lattice._connect_8(self.graph)

    def assign_cost(self, images=[()], alpha=const.IMG_LATTICE_COST_ASSIGNMENT_ALPHA, override=True):
        i = 0
        for n1 in self.graph.nodes():
            for n2 in nx.neighbors(self.graph, n1):
                if self.graph[n1][n2] == {} or override:
                    cost = 0.0
                    # ix = 1
                    for weight, arr in images:
                        i_diff = max(arr[n1[0], n1[1]], arr[n2[0], n2[1]])
                        cost += weight * mth.pow(mth.e, alpha * (i_diff / 255))
                        # graph[n1][n2]['i_diff_' + str(ix)] = i_diff
                        # ix += 1
                        self.graph[n1][n2]['cost'] = cost
