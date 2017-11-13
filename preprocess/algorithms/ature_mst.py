from functools import wraps
from heapq import heappop, heappush
from itertools import count
from multiprocessing import Process

import networkx as nx
import numpy as np

from commons.IMAGE import Image
from commons.timer import check_time

"""
  A minimum spanning tree is a sub-graph of the graph (a tree)
  with the minimum sum of edge weights.  A spanning forest is a
  union of the spanning trees for each connected component of the graph.
  """


@check_time
def job_mst(sub_graph, accumulator_arr):
    sub_mst = nx.algorithms.prim_mst(sub_graph, weight='cost')
    for x, y in sub_mst:
        pass


def _ll_prim_mst(func):
    @wraps(func)
    def inner(image=Image()):

        acc = np.zeros_like(image.img_array[:, :, 1])
        g = image.lattice[0].nodes()
        step = 50000
        num_of_nodes = image.lattice[0].number_of_nodes() - step
        all_p = []

        for i in range(0, num_of_nodes, step):
            p = Process(target=func(sub_graph=nx.subgraph(g, g[i:i + step - 1]), accumulator_arr=acc))
            all_p.append(p)

        for p in all_p:
            p.run()
        for p in all_p:
            p.join()

        return acc

    return inner


@check_time
def prim_mst_edges(graph, weight='cost', data=True):
    if graph.is_directed():
        raise nx.NetworkXError(
            "Minimum spanning tree not defined for directed graphs.")

    push = heappush
    pop = heappop

    nodes = graph.nodes()
    c = count()

    while nodes:
        u = nodes.pop(0)
        frontier = []
        visited = [u]
        for u, v in graph.edges(u):
            push(frontier, (graph[u][v].get(weight, 1), next(c), u, v))

        while frontier:
            _, _, u, v = pop(frontier)
            if v in visited:
                continue
            visited.append(v)
            nodes.remove(v)
            for v, w in graph.edges(v):
                if w not in visited:
                    push(frontier, (graph[v][w].get(weight, 1), next(c), v, w))
            if data:
                yield u, v, graph[u][v]
            else:
                yield u, v


@_ll_prim_mst
def prim_mst(image=Image(), weight='cost'):
    """
     If the graph is not connected a spanning forest is constructed.  A
     spanning forest is a union of the spanning trees for each
     connected component of the graph.
    """

    graph = image.lattice[0]

    t = nx.Graph(nx.prim_mst_edges(graph, weight=weight, data=True))
    # Add isolated nodes
    if len(t) != len(graph):
        t.add_nodes_from([n for n, d in graph.degree().items() if d == 0])
    # Add node and graph attributes as shallow copy
    for n in t:
        t.node[n] = graph.node[n].copy()
    t.graph = graph.graph.copy()
    return t
