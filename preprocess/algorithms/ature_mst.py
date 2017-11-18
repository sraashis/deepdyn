from heapq import heappop, heappush
from itertools import count
from random import shuffle

import networkx as nx
import numpy as np

from commons.timer import check_time

"""
  A minimum spanning tree is a sub-graph of the graph (a tree)
  with the minimum sum of edge weights.  A spanning forest is a
  union of the spanning trees for each connected component of the graph.
  """


@check_time
def _prim_mst(lattice=None, lattice_object=None, weight_limit_per_seed=None, weight=None, seed=None,
              node_limit_per_seed=None, number_of_seeds=None):
    if lattice.is_directed():
        raise nx.NetworkXError(
            "Minimum spanning tree not defined for directed graphs.")

    push = heappush
    pop = heappop

    c = count()
    seed_used = 0
    while seed:
        u = seed.pop(0)
        if seed_used >= number_of_seeds:
            break
        print("Seed: " + str(seed_used))
        seed_used += 1
        node_count = 0
        seed_weight = 0.0
        frontier = []
        visited = [u]
        for u, v in lattice.edges(u):
            push(frontier, (lattice[u][v].get(weight, 1), next(c), u, v))

        while frontier:
            _, _, u, v = pop(frontier)

            if v in visited:
                continue

            # Start with new Seed
            if node_count > node_limit_per_seed or seed_weight > weight_limit_per_seed:
                print('Node Count: ' + str(node_count) + ", Weight: " + str(seed_weight))
                break

            # Keep records and control track
            lattice_object.accumulator[v[0], v[1]] = 255
            node_count += 1

            visited.append(v)

            # We only count non-seed's weight.
            if v in seed:
                seed.remove(v)
            else:
                seed_weight += float(lattice[u][v].get(weight, 1))

            for v, w in lattice.edges(v):
                if w not in visited:
                    push(frontier, (lattice[v][w].get(weight, 1), next(c), v, w))


def run_mst(lattice_object=None, weight_limit_per_seed=20000, weight='cost', seed=None, node_limit_per_seed=10000,
            number_of_seeds=35, expansion_rate=1):
    shuffle(seed)
    lattice_object.accumulator = np.zeros([lattice_object.x_size, lattice_object.y_size], dtype=np.uint8)
    lattice_object.total_weight = 0.0
    _prim_mst(lattice=lattice_object.lattice, lattice_object=lattice_object,
              weight_limit_per_seed=weight_limit_per_seed,
              weight=weight,
              seed=seed, node_limit_per_seed=node_limit_per_seed, number_of_seeds=number_of_seeds)
