import logging as logger

import networkx as nx
import numpy as np


class Lattice:
    def __init__(self, image_2d=None):
        logger.basicConfig(level=logger.INFO)
        self.image_2d = image_2d
        self.grid_size = (2, 3)
        self.k_lattices = []
        self.lattice = None
        self.accumulator = np.zeros_like(image_2d)

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

    def generate_lattice_graph(self, eight_connected=False):
        if eight_connected:
            logger.info(msg='Creating 8-connected lattice.')
        else:
            logger.info(msg='Creating 8-connected lattice.')
        if self.lattice is not None:
            logger.warning(msg='Lattice already exists. Overriding..')
        self.lattice = nx.grid_graph([self.image_2d.shape[0], self.image_2d.shape[1]])

        if eight_connected:
            Lattice._connect_8(self.lattice)

    @staticmethod
    def _get_sub_lattice(i, j, x_block_size, y_block_size):
        for p in range(i, i + x_block_size, 1):
            for q in range(j, j + y_block_size, 1):
                yield (p, q)

    @staticmethod
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
                logger.info(msg=str(i) + ',' + str(j))
                sub_graphs_nodes.append(nx.subgraph(full_lattice, Lattice._get_sub_lattice(i, j, x_size, y_size)))
        return sub_graphs_nodes

    @staticmethod
    def assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0)), metrics_name=None):
        for i, j in graph.nodes():
            graph[(i, j)][metrics_name] = metrics[i, j]

    @staticmethod
    def get_lattice_portion(image_2d=None, a_lattice=nx.Graph()):
        res = np.zeros_like(image_2d)
        for i, j in a_lattice.nodes():
            res[i, j] = image_2d[i, j]
        return res
