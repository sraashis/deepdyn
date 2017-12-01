from heapq import heappop, heappush
from itertools import count
from random import shuffle

import numpy as np

from commons.timer import check_time


@check_time
def _prim_mst(lattice_object=None,
              weight=None,
              seed=None,
              threshold=None):
    push = heappush
    pop = heappop

    c = count()
    while seed:
        u = seed.pop(0)
        frontier = []
        visited = [u]
        for u, v in lattice_object.lattice.edges(u):
            push(frontier, (lattice_object.lattice[u][v].get(weight, 1), next(c), u, v))

        while frontier:
            _, _, u, v = pop(frontier)

            if v in visited:
                continue

            if lattice_object.lattice[u][v].get(weight, 1) < threshold:
                lattice_object.accumulator[v[0], v[1]] = 255
            else:
                break

            visited.append(v)

            if v in seed:
                seed.remove(v)

            for v, w in lattice_object.lattice.edges(v):
                if w not in visited:
                    push(frontier, (lattice_object.lattice[v][w].get(weight, 1), next(c), v, w))


@check_time
def _dijkstra(lattice_object=None,
              weight_limit_per_seed=None,
              weight=None,
              seed=None,
              number_of_seeds=None):
    push = heappush
    pop = heappop

    c = count()
    seed_used = 0
    while seed:
        u = seed.pop(0)
        if seed_used >= number_of_seeds:
            break
        seed_used += 1
        seed_weight = 0.0
        frontier = []
        visited = [u]
        for u, v in lattice_object.lattice.edges(u):
            push(frontier, (lattice_object.lattice[u][v].get(weight, 1), next(c), u, v))

        while frontier:
            _, _, u, v = pop(frontier)

            if v in visited:
                continue

            if seed_weight > weight_limit_per_seed:
                break

            lattice_object.accumulator[v[0], v[1]] = 255
            node_count += 1
            seed_weight += float(lattice_object.lattice[u][v].get(weight, 1))

            visited.append(v)

            if v in seed:
                seed.remove(v)
                node_count = 0
                seed_weight = 0.0

            for v, w in lattice_object.lattice.edges(v):
                if w not in visited:
                    push(frontier, (lattice_object.lattice[v][w].get(weight, 1), next(c), v, w))


def run_mst(lattice_object=None,
            weight='cost',
            seed=None, threshold=3.0):
    shuffle(seed)
    lattice_object.accumulator = np.zeros([lattice_object.x_size, lattice_object.y_size], dtype=np.uint8)
    lattice_object.total_weight = 0.0
    _prim_mst(lattice_object=lattice_object,
              weight=weight,
              seed=seed, threshold=threshold)


def run_dijkstra(lattice_object=None,
                 weight_limit_per_seed=20000,
                 weight='cost',
                 seed=None,
                 number_of_seeds=10):
    shuffle(seed)
    lattice_object.accumulator = np.zeros([lattice_object.x_size, lattice_object.y_size], dtype=np.uint8)
    lattice_object.total_weight = 0.0
    _dijkstra(lattice_object=lattice_object,
              weight_limit_per_seed=weight_limit_per_seed,
              weight=weight,
              seed=seed,
              number_of_seeds=number_of_seeds)
