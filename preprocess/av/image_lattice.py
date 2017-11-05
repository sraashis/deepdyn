import networkx as nx


def connect_eight_nodes_in_lattice_graph(graph):
    for i, j in graph.nodes():
        n0 = (i, j)
        n1 = (i - 1, j + 1)
        n2 = (i + 1, j - 1)
        n3 = (i - 1, j - 1)
        n4 = (i + 1, j + 1)
        if n1 in graph.nodes():
            graph.add_edge(n0, n1)
        if n2 in graph.nodes():
            graph.add_edge(n0, n2)
        if n3 in graph.nodes():
            graph.add_edge(n0, n3)
        if n4 in graph.nodes():
            graph.add_edge(n0, n4)


def create_weighted_lattice_graph(graph, image_arr, gabor_arr):
    graph = nx.grid_graph([image_arr.shape[0], image_arr.shape[1]])
    n_pos = dict(zip(graph.nodes(), graph.nodes()))
    connect_eight_nodes_in_lattice_graph(graph)
    return graph, n_pos
