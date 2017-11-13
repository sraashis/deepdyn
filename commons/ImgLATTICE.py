import networkx as nx
import numpy as np

import preprocess.av.lattice_utils as lat
from commons.LOGGER import Logger
from commons.timer import check_time


class Lattice(Logger):
    def __init__(self, image_2d=[]):
        Logger.__init__()
        self.image_2d = image_2d
        self.grid_size = (2, 3)
        self.k_lattices = []
        self.lattice = None
        self.accumulator = np.zeros_like(image_2d)

    @check_time
    def generate_lattice_graph(self):
        Logger.log('Creating 4-connected lattice.')
        if self.lattice is not None:
            Logger.warn('Lattice already exists. Overriding..')
        self.lattice = lat.generate_lattice_graph(self.image_2d)

    @check_time
    def chunk_lattice(self, full_lattice, grid_size):
        Logger.log("Chunking lattice to " + str(grid_size) + " Lattices.")
        self.k_lattices = lat.chunk_lattice(self.image_2d, full_lattice, grid_size)

    @staticmethod
    @check_time
    def assign_cost(graph=nx.Graph(), images=[()], alpha=10, override=False, log=True):
        Logger.log('Calculating cost of moving to a neighbor.')
        if override:
            Logger.warn("Overriding..")
        lat.assign_cost(graph, images=images, alpha=alpha, override=override, log=log)

    @staticmethod
    @check_time
    def assign_node_metrics(graph=nx.Graph(), metrics=np.ndarray((0, 0))):
        lat.assign_node_metrics(graph=graph, metrics=metrics)
