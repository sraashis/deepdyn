from heapq import heappop, heappush
from itertools import count
from multiprocessing import Process
import numpy as np

import networkx as nx

from commons.timer import check_time

"""
  A minimum spanning tree is a sub-graph of the graph (a tree)
  with the minimum sum of edge weights.  A spanning forest is a
  union of the spanning trees for each connected component of the graph.
  """


@check_time
def _prim_mst_edges(lattice=None, lattice_object=None, threshold=None, weight=None, seed=None):
    if lattice.is_directed():
        raise nx.NetworkXError(
            "Minimum spanning tree not defined for directed graphs.")

    push = heappush
    pop = heappop

    nodes = lattice.nodes()
    c = count()

    while seed:
        u = seed.pop(0)
        frontier = []
        visited = [u]
        for u, v in lattice.edges(u):
            push(frontier, (lattice[u][v].get(weight, 1), next(c), u, v))

        while frontier:
            _, _, u, v = pop(frontier)
            if v in visited:
                continue
            visited.append(v)
            nodes.remove(v)
            for v, w in lattice.edges(v):
                if w not in visited:
                    push(frontier, (lattice[v][w].get(weight, 1), next(c), v, w))

            if 0 == threshold:
                return

            lattice_object.accumulator[v[0], v[1]] = 255
            lattice_object.total_weight += float(lattice[u][v].get(weight, 1))
            threshold = threshold - 1


def _prim_mst(lattice=None, lattice_object=None, threshold=None, weight=None, seed=None):
    """
     If the graph is not connected a spanning forest is constructed.  A
     spanning forest is a union of the spanning trees for each
     connected component of the graph.
    """
    # Reset before running again.
    lattice_object.accumulator = np.zeros([lattice_object.x_size, lattice_object.y_size], dtype=np.uint8)
    lattice_object.total_weight = 0.0
    _prim_mst_edges(lattice=lattice, lattice_object=lattice_object, threshold=threshold, weight=weight, seed=seed)


def run_mst(lattice_object=None, threshold=None, weight='cost', seed=None):
    _prim_mst(lattice=lattice_object.lattice, lattice_object=lattice_object, threshold=threshold, weight=weight,
              seed=seed)


def run_mst_parallel(lattice_object=None, threshold=None, weight='cost', seed=None, test_index=-1):
    all_p = []
    if test_index >= 0:
        seed = set.intersection(set(seed), lattice_object.k_lattices[test_index].nodes())
        p = Process(
            target=_prim_mst(lattice=lattice_object.k_lattices[test_index], lattice_object=lattice_object,
                             threshold=threshold, weight=weight,
                             seed=list(seed)))
        all_p.append(p)
    else:
        for a_lattice in lattice_object.k_lattices:
            p = Process(
                target=_prim_mst(lattice=a_lattice, lattice_object=lattice_object, threshold=threshold, weight=weight,
                                 seed=seed))
        all_p.append(p)

    for p in all_p:
        p.start()

    for p in all_p:
        p.join()
