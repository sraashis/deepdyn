import matplotlib.pyplot as plt
import networkx as nx


def show_graph(auxiliary_matrix, node_pos=None):
    graph = nx.from_scipy_sparse_matrix(auxiliary_matrix)
    nx.draw_networkx(graph, pos=node_pos, edge_color='blue', node_color='red', with_labels=False, node_size=2.5,
                     width=0.5)
    plt.show()
