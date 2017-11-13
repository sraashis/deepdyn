import networkx as nx

import preprocess.av.lattice_utils as lat
from commons.LOGGER import Logger
from commons.timer import check_time
import numpy as np


class Lattice:
    def __init__(self, image_2d=None, lattice_grid_size=(1, 1)):
        self.image_2d = image_2d
        self.grid_size = lattice_grid_size
        self.k_lattices = []

    @check_time
    def create_lattice_graph(self, image_arr_2d=None):
        self.log('Creating 4-connected lattice.')
        if self.lattice is not None:
            self.warn('Lattice already exists. Overriding..')
        self.k_lattices = lat.create_lattice_graph(image_arr_2d)

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
