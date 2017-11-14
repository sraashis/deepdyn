import math as mth
from multiprocessing import Process

import networkx as nx


def assign_cost(graph=nx.Graph(), images=[()], alpha=10, override=False, log=False):
    i = 0
    for n1 in graph.nodes():
        for n2 in nx.neighbors(graph, n1):
            if graph[n1][n2] == {} or override:
                cost = 0.0
                ix = 1
                for weight, arr in images:
                    i_diff = abs(float(arr[n1[0], n1[1]]) - float(arr[n2[0], n2[1]]))
                    cost += weight * mth.pow(mth.e, alpha * (i_diff / 255))
                    graph[n1][n2]['i_diff_' + str(ix)] = i_diff
                    ix += 1
                graph[n1][n2]['cost'] = cost
        if log:
            print('\r' + str(i) + ': ' + str(n1), end='')
            i += 1


def assign_cost_parallel(lattices, images=[()], alpha=10, override=False):
    all_p = []
    for a_lattice in lattices:
        p = Process(target=assign_cost(a_lattice, images, alpha, override, False))
        all_p.append(p)
    for p in all_p:
        p.run()
        p.join()
