import sys
from pprint import pprint
from collections import deque
from queue import Queue, PriorityQueue


class AStarGraph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.h_values = dict([(k, 1) for k in adjacency_list])

    def get_neighbours(self, node):
        return self.adjacency_list[node]

    def h(self, node):
        return self.h_values[node]

    def a_star(self, start_node, stop_node):
        # open_list is a list of nodes which have been visited, but who's neighbors
        # haven't all been inspected, starts off with the start node
        # closed_list is a list of nodes which have been visited
        # and who's neighbors have been inspected
        open_list = set([start_node])
        closed_list = set([])

        # g contains current distances from start_node to all other nodes
        # the default value (if it's not found in the map) is +infinity
        g = {}

        g[start_node] = 0

        # parents contains an adjacency map of all nodes
        parents = {}
        parents[start_node] = start_node

        while len(open_list) > 0:
            n = None

            # find a node with the lowest value of f() - evaluation function
            for v in open_list:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v

            if n == None:
                print('Path does not exist!')
                return None

            # if the current node is the stop_node
            # then we begin reconstructin the path from it to the start_node
            if n == stop_node:
                reconst_path = []

                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]

                reconst_path.append(start_node)

                reconst_path.reverse()

                print('Path found: {}'.format(reconst_path))

                return reconst_path

            # for all neighbors of the current node do
            for (m, weight) in self.get_neighbours(n):
                # if the current node isn't in both open_list and closed_list
                # add it to open_list and note n as it's parent
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight

                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closed_list, move it to open_list
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)

            # remove n from the open_list, and add it to closed_list
            # because all of his neighbors were inspected
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return None


class DijkstraGraph:
    def __init__(self, num_of_nodes, directed=True):
        self._num_of_nodes = num_of_nodes
        self._directed = directed
        self._edges = [
                [-1 for i in range(num_of_nodes)]
                for j
                in range(num_of_nodes)
            ]
        self._visited = []


    def add_edge(self, u, v, weight):
        self._edges[u][v] = weight
        self._edges[v][u] = weight

    # Methods for removing edges

    # Methods for searching a graph
        # BFS, DFS, Dijkstra, A*...
    ## Dijkstra's algorithm
    def dijkstra(self, start_node):
        shortest_paths = {v: float('inf') for v in range(self._num_of_nodes)}
        shortest_paths[start_node] = 0

        queue = PriorityQueue()
        queue.put((0, start_node))

        while not queue.empty():
            (dist, current_node) = queue.get()
            self._visited.append(current_node)

            for neighbour in range(self._num_of_nodes):
                if self._edges[current_node][neighbour] != -1:
                    distance = self._edges[current_node][neighbour]

                    if neighbour not in self._visited:
                        old_cost = shortest_paths[neighbour]
                        new_cost = shortest_paths[current_node] + distance

                        if new_cost < old_cost:
                            queue.put((new_cost, neighbour))
                            shortest_paths[neighbour] = new_cost

        return shortest_paths

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_representation(self):
        num_of_edges = len(self._edges)

        for i in range(num_of_edges):
            print("edge ", i+1, ": ", self._edges[i])


class ListOfEdgesGraph:
    def __init__(self, num_of_nodes, directed=True):
        self._num_of_nodes = num_of_nodes
        self._directed = directed
        self._edges = []
        self._visited = []

    def add_edge(self, node1, node2, weight=1):
        self._edges.append([node1, node2, weight])

        if not self._directed:
            self._edges.append([node2, node1, weight])

    # Methods for removing edges

    # Methods for searching a graph
        # BFS, DFS, Dijkstra, A*...

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_representation(self):
        num_of_edges = len(self._edges)

        for i in range(num_of_edges):
            print("edge ", i+1, ": ", self._edges[i])


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

    def print_representation(self):
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

    # Methods for traversing a graph
    ## Breadth-first traversal
    def bft(self, start_node):
        visited = set()
        queue = Queue()
        queue.put(start_node)
        visited.add(start_node)

        while not queue.empty():
            current_node = queue.get()
            print(current_node, end = " ")

            for (next_node, weight) in self._adjacency_list[current_node]:
                if next_node not in visited:
                    queue.put(next_node)
                    visited.add(next_node)

    # Methods for searching a graph
    ## Breadth-first search
    def bfs(self, start_node, target_node):
        visited = set()
        queue = Queue()

        # Add the start node to the queue and visited list
        queue.put(start_node)
        visited.add(start_node)

        # start_node has no parents
        parent_map = dict()
        parent_map[start_node] = None

        path_found = False

        while not queue.empty():
            current_node = queue.get()

            if current_node == target_node:
                path_found = True
                break

            for (next_node, weight) in self._adjacency_list[current_node]:
                if next_node not in visited:
                    queue.put(next_node)
                    parent_map[next_node] = current_node
                    visited.add(next_node)

        path = []

        if path_found:
            path.append(target_node)

            while parent_map[target_node] is not None:
                path.append(parent_map[target_node])
                target_node = parent_map[target_node]

            path.reverse()

        return path

    ## Depth-first search
    def dfs(self, start, target, path=[], visited=set()):
        path.append(start)
        visited.add(start)

        if start == target:
            return path

        for (neighbour, weight) in self._adjacency_list[start]:
            if neighbour not in visited:
                result = self.dfs(neighbour, target, path, visited)

                if result is not None:
                    return result

        path.pop()

        return None

    # Methods for finding a minimum spanning tree
        # Prim's algorithm, Kruskal's algorithm, Borůvka's algorithm...

    def print_representation(self):
        for key in self._adjacency_list:
            print("node", key, ": ", self._adjacency_list[key])


def make_graph(representation, num_of_nodes, directed=True):
    graph_types_map = {
        "list of edges": ListOfEdgesGraph,
        "adjacency matrix": AdjacencyMatrixGraph,
        "adjacency list": AdjacencyListGraph,
        "dijkstra": DijkstraGraph,
    }

    if representation not in graph_types_map:
        print(f'Representation "{representation}" is not implemented.')
        print(f'Available representations are: {", ".join([t for t in graph_types_map])}')
        sys.exit()

    return graph_types_map[representation](num_of_nodes, directed)


def show_representation_example():
    graph = make_graph("list of edges", 5)

    graph.add_edge(0, 0, 25)
    graph.add_edge(0, 1, 5)
    graph.add_edge(0, 2, 3)
    graph.add_edge(1, 3, 1)
    graph.add_edge(1, 4, 15)
    graph.add_edge(4, 2, 7)
    graph.add_edge(4, 3, 11)

    print("\nTesting representations:\n")
    graph.print_representation()
    print()


def show_bft_example():
    graph = make_graph("adjacency list", 5, directed=False)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 2)
    graph.add_edge(1, 4)
    graph.add_edge(2, 3)

    print("\nTesting BFT:\n")
    graph.print_representation()
    print(f"\nTraversal using bft:")
    graph.bft(0)
    print()


def show_bfs_example():
    graph = make_graph("adjacency list", 6, directed=False)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(0, 4)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(2, 5)
    graph.add_edge(3, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)

    print("\nTesting BFS:\n")
    graph.print_representation()
    print()
    path = graph.bfs(0, 5)
    print("Path from node 0 to node 5 is:", path)
    print()


def show_dfs_example():
    graph = make_graph("adjacency list", 5, directed=False)

    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)


    print("\nTesting DFS:\n")
    graph.print_representation()
    print()
    path = graph.dfs(0, 3)
    print("Path from node 0 to node 3 is:", path)
    print()


def show_dijkstra_example():
    graph = make_graph('dijkstra', 9, directed=False)
    graph.add_edge(0, 1, 4)
    graph.add_edge(0, 6, 7)
    graph.add_edge(1, 6, 11)
    graph.add_edge(1, 7, 20)
    graph.add_edge(1, 2, 9)
    graph.add_edge(2, 3, 6)
    graph.add_edge(2, 4, 2)
    graph.add_edge(3, 4, 10)
    graph.add_edge(3, 5, 5)
    graph.add_edge(4, 5, 15)
    graph.add_edge(4, 7, 1)
    graph.add_edge(4, 8, 5)
    graph.add_edge(5, 8, 12)
    graph.add_edge(6, 7, 1)
    graph.add_edge(7, 8, 3)

    print("\nTesting Dijkstra's algorithm:\n")
    graph.print_representation()
    print()
    path = graph.dijkstra(0)
    for n in range(len(path)):
        print(f"Distance from node 0 to node {n} is {path[n]}")
    print()


def show_a_star_example():
    adjacency_list = {
        'A': [('B', 1), ('C', 3), ('D', 7)],
        'B': [('D', 5)],
        'C': [('D', 12)],
        'D': []
    }

    graph = AStarGraph(adjacency_list)

    print("\nTesting the A* algorithm:\n")
    graph.a_star('A', 'D')
    print()


if __name__ == '__main__':
    show_representation_example()
    show_bft_example()
    show_bfs_example()
    show_dfs_example()
    show_dijkstra_example()
    show_a_star_example()
