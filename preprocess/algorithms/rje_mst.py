from heapq import heappop, heappush
from operator import itemgetter
from itertools import count
from math import isnan

import networkx as nx
from networkx.utils import UnionFind, not_implemented_for

# def prim_mst_edges(G, minimum=True, weight='weight',
#                    keys=True, data=True, ignore_nan=False):
#     is_multigraph = G.is_multigraph()
#     push = heappush
#     pop = heappop

#     nodes = list(G)
#     c = count()

#     sign = 1 if minimum else -1

#     while nodes:
#         u = nodes.pop(0)
#         frontier = []
#         visited = [u]
#         if is_multigraph:
#             for v, keydict in G.adj[u].items():
#                 for k, d in keydict.items():
#                     wt = d.get(weight, 1) * sign
#                     if isnan(wt):
#                         if ignore_nan:
#                             continue
#                         msg = "NaN found as an edge weight. Edge %s"
#                         raise ValueError(msg % ((u, v, k, d),))
#                     push(frontier, (wt, next(c), u, v, k, d))
#         else:
#             for v, d in G.adj[u].items():
#                 wt = d.get(weight, 1) * sign
#                 if isnan(wt):
#                     if ignore_nan:
#                         continue
#                     msg = "NaN found as an edge weight. Edge %s"
#                     raise ValueError(msg % ((u, v, d),))
#                 push(frontier, (wt, next(c), u, v, d))
#         while frontier:
#             if is_multigraph:
#                 W, _, u, v, k, d = pop(frontier)
#             else:
#                 W, _, u, v, d = pop(frontier)
#             if v in visited:
#                 continue
#             # Multigraphs need to handle edge keys in addition to edge data.
#             if is_multigraph and keys:
#                 if data:
#                     yield u, v, k, d
#                 else:
#                     yield u, v, k
#             else:
#                 if data:
#                     yield u, v, d
#                 else:
#                     yield u, v
#             # update frontier
#             visited.append(v)
#             nodes.remove(v)
#             if is_multigraph:
#                 for w, keydict in G.adj[v].items():
#                     if w in visited:
#                         continue
#                     for k2, d2 in keydict.items():
#                         new_weight = d2.get(weight, 1) * sign
#                         push(frontier, (new_weight, next(c), v, w, k2, d2))
#             else:
#                 for w, d2 in G.adj[v].items():
#                     if w in visited:
#                         continue
#                     new_weight = d2.get(weight, 1) * sign
#                     push(frontier, (new_weight, next(c), v, w, d2))


def prim_mst_edges(G, minimum=True, weight='weight',
                   keys=True, data=True, ignore_nan=False):
    push = heappush
    pop = heappop

    nodes = list(G)
    c = count()

    sign = 1 if minimum else -1

    while nodes:
        u = nodes.pop(0)
        frontier = []
        visited = [u]
        for v, d in G.adj[u].items():
            wt = d.get(weight, 1) * sign
            if isnan(wt):
                if ignore_nan:
                    continue
                msg = "NaN found as an edge weight. Edge %s"
                raise ValueError(msg % ((u, v, d),))
            push(frontier, (wt, next(c), u, v, d))

        while frontier:
            W, _, u, v, d = pop(frontier)
            if v in visited:
                continue
            if data:
                yield u, v, d
            else:
                yield u, v
            # update frontier
            visited.append(v)
            nodes.remove(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get(weight, 1) * sign
                push(frontier, (new_weight, next(c), v, w, d2))


def kruskal_mst_edges(G, minimum=True, weight='weight',
                      keys=True, data=True, ignore_nan=False):
    subtrees = UnionFind()
    edges = G.edges(data=True)

    def filter_nan_edges(edges=edges, weight=weight):
        sign = 1 if minimum else -1
        for u, v, d in edges:
            wt = d.get(weight, 1) * sign
            if isnan(wt):
                if ignore_nan:
                    continue
                msg = "NaN found as an edge weight. Edge %s"
                raise ValueError(msg % ((u, v, d),))
            yield wt, u, v, d
    edges = sorted(filter_nan_edges(), key=itemgetter(0))

    for wt, u, v, d in edges:
        if subtrees[u] != subtrees[v]:
            if data:
                yield (u, v, d)
            else:
                yield (u, v)
            subtrees.union(u, v)


def minimum_spanning_tree(G, weight='weight', ignore_nan=False):
    # edges = prim_mst_edges(G, minimum=True, weight=weight, keys=True,
    #                                data=True, ignore_nan=ignore_nan)
    edges = kruskal_mst_edges(G, minimum=True, weight=weight, keys=True,
                                   data=True, ignore_nan=ignore_nan)
    T = G.fresh_copy()  # Same graph class as G
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(edges)
    return T

def valid_conn_comps(G,seed_nodes):    
    T = G.fresh_copy()
    T.graph.update(G.graph)
    T.add_nodes_from(G.nodes.items())
    T.add_edges_from(G.edges())

    seed_set = set(seed_nodes)
    nodesToRemove = []
    for comp in nx.connected_components(G):
        if seed_set.isdisjoint(comp):
            nodesToRemove.extend(comp)
    T.remove_nodes_from(nodesToRemove)
    return T, nodesToRemove


