from heapq import heappop, heappush
from itertools import count
from multiprocessing import Process

import networkx as nx
import numpy as np

from commons.timer import check_time
from commons.ImgLATTICE import Lattice

"""
  A minimum spanning tree is a sub-graph of the graph (a tree)
  with the minimum sum of edge weights.  A spanning forest is a
  union of the spanning trees for each connected component of the graph.
  """


@check_time
def _prim_mst_edges(graph, weight='cost', data=True, acc=None):
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


def _prim_mst(graph, weight='cost', acc=None):
    """
     If the graph is not connected a spanning forest is constructed.  A
     spanning forest is a union of the spanning trees for each
     connected component of the graph.
    """

    t = nx.Graph(_prim_mst_edges(graph, weight=weight, data=True))
    # Add isolated nodes
    if len(t) != len(graph):
        t.add_nodes_from([n for n, d in graph.degree().items() if d == 0])
    # Add node and graph attributes as shallow copy
    for n in t:
        t.node[n] = graph.node[n].copy()
    t.graph = graph.graph.copy()
    return t


def run_mst(img_lattice=Lattice()):
    all_p = []
    for a_lattice in img_lattice.k_lattices:
        p = Process(target=_prim_mst(a_lattice, img_lattice.accumulator))
        all_p.append(p)
    for p in all_p:
        p.run()
    for p in all_p:
        p.join()
