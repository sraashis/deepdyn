import math as mth

import networkx as nx

import commons.constants as const
from commons.timer import checktime


@checktime
def run_segmentation(accumulator_2d=None, images_used=None,
                     seed_list=None,
                     params=None, graph=None):

    edges_to_delete = []
    for e in graph.edges():
        cost = 0.0
        for weight, arr in images_used:
            i_diff = max(arr[e[0]], arr[e[1]])
            cost += weight * mth.pow(mth.e, params['alpha'] * (i_diff / 255))
            graph[e[0]][e[1]][const.GRAPH_WEIGHT_METRICS] = cost
        if cost > params['seg_threshold']:
            edges_to_delete.append(e)

    graph.remove_edges_from(edges_to_delete)

    for component in nx.connected_components(graph):
        if component.isdisjoint(seed_list) is False:
            for node in component:
                accumulator_2d[node] = 255
