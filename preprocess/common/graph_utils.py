import networkx as nx


def show_graph(auxiliary_matrix, node_pos=None, node_color='red',edge_color='black'):
    graph = nx.from_scipy_sparse_matrix(auxiliary_matrix)
    nx.draw_networkx(graph, pos=node_pos, edge_color=edge_color, node_color=node_color, with_labels=False, node_size=4,
                     width=0.5)


def color_artery(x): return x == 1 and 'r' or 'b'


def color_vein(x): return x == 1 and 'b' or 'r'


def color_av(a, v):
    if a == 1 and v == 1:
        return 'g'
    if a == 1 and v == 0:
        return 'b'
    if a == 0 and v == 1:
        return 'r'
    return 'g'
