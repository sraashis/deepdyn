from heapq import heappop, heappush
from itertools import count

import networkx as nx

"""
  A minimum spanning tree is a sub-graph of the graph (a tree)
  with the minimum sum of edge weights.  A spanning forest is a
  union of the spanning trees for each connected component of the graph.
  """


def get_skeleton(nodes, metrics):
    while nodes:
        n = nodes.pop(0)
        if n[metrics] == 0:
            yield n


def prim_mst_edges(graph, weight='weight', data=True):
    if graph.is_directed():
        raise nx.NetworkXError(
            "Minimum spanning tree not defined for directed graphs.")

    push = heappush
    pop = heappop

    nodes = graph.nodes()
    c = count()

    nodes = get_skeleton(graph, weight)

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


def prim_mst(graph, weight='weight'):
    """
     If the graph is not connected a spanning forest is constructed.  A
     spanning forest is a union of the spanning trees for each
     connected component of the graph.
    """

    t = nx.Graph(nx.prim_mst_edges(graph, weight=weight, data=True))
    # Add isolated nodes
    if len(t) != len(graph):
        t.add_nodes_from([n for n, d in graph.degree().items() if d == 0])
    # Add node and graph attributes as shallow copy
    for n in t:
        t.node[n] = graph.node[n].copy()
    t.graph = graph.graph.copy()
    return t
