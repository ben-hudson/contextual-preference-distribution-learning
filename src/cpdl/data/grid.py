import networkx as nx


def grid_graph(w: int, h: int, bidirectional: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()
    for row in range(h):
        for col in range(w):
            if row < h - 1:
                G.add_edge((row, col), (row + 1, col))
                if bidirectional:
                    G.add_edge((row + 1, col), (row, col))
            if col < w - 1:
                G.add_edge((row, col), (row, col + 1))
                if bidirectional:
                    G.add_edge((row, col + 1), (row, col))
    return G
