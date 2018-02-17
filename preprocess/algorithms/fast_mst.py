import math as mth

import networkx as nx
import numpy as np

import commons.constants as const


def run_segmentation(accumulator=None,
                     seed_list=None,
                     segmentation_threshold=const.SEGMENTATION_THRESHOLD,
                     alpha=const.IMG_LATTICE_COST_ASSIGNMENT_ALPHA,
                     img_gabor_contribution=const.IMG_LATTICE_COST_GABOR_IMAGE_CONTRIBUTION,
                     img_original_contribution=const.IMG_LATTICE_COST_ORIGINAL_IMAGE_CONTRIBUTION):
    graph = accumulator.img_obj.graph.copy()
    img_used = [(img_gabor_contribution, accumulator.img_obj.img_gabor),
                (img_original_contribution, accumulator.img_obj.img_array)]

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
                accumulator.accumulator[node] = 255

    return graph
