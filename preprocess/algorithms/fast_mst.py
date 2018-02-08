import commons.constants as const
from commons.timer import check_time
import math as mth
import networkx as nx
import numpy as np


def run_segmentation(image_object=None,
                     lattice_object=None,
                     seed_list=None,
                     segmentation_threshold=const.SEGMENTATION_THRESHOLD,
                     alpha=const.IMG_LATTICE_COST_ASSIGNMENT_ALPHA,
                     img_gabor_contribution=const.IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION,
                     img_original_contribution=const.IMG_LATTICE_COST_ORIGINAL_IMAGE_CONTRIBUTION):

    graph = lattice_object.graph.copy()
    lattice_object.accumulator = np.zeros([lattice_object.x_size, lattice_object.y_size], dtype=np.uint8)
    img_used = [(img_gabor_contribution, image_object.img_gabor), (img_original_contribution, image_object.img_array)]

    edges_to_delete = []
    for e in graph.edges():
        cost = 0.0
        for weight, arr in img_used:
            i_diff = max(arr[e[0]], arr[e[1]])
            cost += weight * mth.pow(mth.e, alpha * (i_diff / 255))
            graph[e[0]][e[1]][const.GRAPH_WEIGHT_METRICS] = cost
        if cost > segmentation_threshold:
            edges_to_delete.append(e)

    graph.remove_edges_from(edges_to_delete)

    for component in nx.connected_components(graph):
        if component.isdisjoint(seed_list) is False:
            for node in component:
                lattice_object.accumulator[node] = 255

    return graph
