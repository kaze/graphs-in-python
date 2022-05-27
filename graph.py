import sys
from pprint import pprint

class ListOfEdgesGraph:
    def __init__(self, num_of_nodes, directed=True):
        self._num_of_nodes = num_of_nodes
        self._directed = directed
        self._list_of_edges = []

    def add_edge(self, node1, node2, weight=1):
        self._list_of_edges.append([node1, node2, weight])

        if not self._directed:
            self._list_of_edges.append([node2, node1, weight])

    # Methods for removing edges

    # Methods for searching a graph
        # BFS, DFS, Dijkstra, A*...

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_graph(self):
        num_of_edges = len(self._list_of_edges)

        for i in range(num_of_edges):
            print("edge ", i+1, ": ", self._list_of_edges[i])


class AdjacencyMatrixGraph:
    def __init__(self, num_of_nodes, directed=True):
        self._num_of_nodes = num_of_nodes
        self._directed = directed
        self._adjacency_matrix = [
            [0 for column in range(num_of_nodes)]
            for row in range(num_of_nodes)
        ]

    def add_edge(self, node1, node2, weight=1):
        self._adjacency_matrix[node1][node2] = weight

        if not self._directed:
            self._adjacency_matrix[node2][node1] = weight

    # Methods for removing edges

    # Methods for searching a graph
        # BFS, DFS, Dijkstra, A*...

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_graph(self):
        pprint(self._adjacency_matrix)


class AdjacencyListGraph:
    def __init__(self, num_of_nodes, directed=True):
        self._num_of_nodes = num_of_nodes
        self._directed = directed
        self._nodes = range(self._num_of_nodes)
        self._adjacency_list = {node: set() for node in self._nodes}

    def add_edge(self, node1, node2, weight=1):
        self._adjacency_list[node1].add((node2, weight))

        if not self._directed:
            self._adjacency_list[node2].add((node1, weight))

    # Methods for removing edges

    # Methods for searching a graph
        # BFS, DFS, Dijkstra, A*...

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_graph(self):
        for key in self._adjacency_list:
            print("node", key, ": ", self._adjacency_list[key])


def make_graph(representation, num_of_nodes, directed=True):
    graph_types_map = {
        "list of edges": ListOfEdgesGraph,
        "adjacency matrix": AdjacencyMatrixGraph,
        "adjacency list": AdjacencyListGraph
    }

    if representation not in graph_types_map:
        print(f'Representation "{representation}" is not implemented.')
        print(f'Available representations are: {", ".join([t for t in graph_types_map])}')
        sys.exit()

    return graph_types_map[representation](num_of_nodes, directed)


if __name__ == '__main__':

    graph = make_graph("adjacency list", 5)

    graph.add_edge(0, 0, 25)
    graph.add_edge(0, 1, 5)
    graph.add_edge(0, 2, 3)
    graph.add_edge(1, 3, 1)
    graph.add_edge(1, 4, 15)
    graph.add_edge(4, 2, 7)
    graph.add_edge(4, 3, 11)

    graph.print_graph()
