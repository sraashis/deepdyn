import logging as logger

import networkx as nx
import numpy as np


class Lattice:
    def __init__(self, image_arr_2d):
        logger.basicConfig(level=logger.INFO)
        self.x_size, self.y_size = image_arr_2d.shape
        self.k_lattices = []
        self.lattice = None
        self.accumulator = np.zeros([self.x_size, self.y_size])

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
            logger.info(msg='Creating 4-connected lattice.')
        if self.lattice is not None:
            logger.warning(msg='Lattice already exists. Overriding..')
        self.lattice = nx.grid_graph([self.x_size, self.y_size])

        if eight_connected:
            Lattice._connect_8(self.lattice)

    @staticmethod
    def _get_sub_lattice(i, j, x_block_size, y_block_size):
        for p in range(i, i + x_block_size, 1):
            for q in range(j, j + y_block_size, 1):
                yield (p, q)

    def chunk_lattice(self, full_lattice, chunk_size=(0, 0)):
        self.k_lattices = []
        x_block_size = int(self.x_size / chunk_size[0])
        y_block_size = int(self.y_size / chunk_size[1])

        remain_x = self.x_size % x_block_size
        x_end = self.x_size - remain_x
        remain_y = self.y_size % y_block_size
        y_end = self.y_size - remain_y

        for i in range(0, x_end, x_block_size):
            for j in range(0, y_end, y_block_size):
                x_size = x_block_size
                y_size = y_block_size
                if i + x_block_size == x_end:
                    x_size = x_block_size + remain_x
                if j + y_block_size == y_end:
                    y_size = y_block_size + remain_y
                logger.info(msg=str(i) + ',' + str(j))
                self.k_lattices.append(nx.subgraph(full_lattice, Lattice._get_sub_lattice(i, j, x_size, y_size)))

    @staticmethod
    def get_slice_focused(image_array_2d=None, a_lattice=nx.Graph()):
        res = np.full(image_array_2d.shape, 200, dtype=np.uint8)
        for i, j in a_lattice.nodes():
            res[i, j] = image_array_2d[i, j]
        return res

    @staticmethod
    def get_slice_only(image_array_2d=None, a_lattice=nx.Graph()):
        x_min, y_min = min(a_lattice.nodes())
        x_max, y_max = max(a_lattice.nodes())
        return image_array_2d[x_min:x_max + 1, y_min:y_max + 1]
