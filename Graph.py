import numpy as np
from random import random, randint
from copy import deepcopy
from queue import PriorityQueue


class Graph:
	def __init__(self, graph=None):
		if graph is None:
			graph = {}
		self.graph = graph
		self.weighted = False

	# dict initializer
	@classmethod
	def from_dict(cls, graph):
		return cls(graph)

	# array initializer
	@classmethod
	def from_array(cls, graph: np.array, nodes: list = None):
		if nodes is None:
			nodes = [*range(1, len(graph) + 1)]
		return cls.from_dict(
			cls._array_to_dict(graph, nodes)
		)

	@staticmethod
	def from_edges(filename: str, directed = 0):
		"""
		Generates a graphs from a text file, where each line defines one edge.\n
		Filename is a file path.
		"""
		graph = Graph()
		file = open(filename, "r")
		for line in file:
			words = line.strip().split()
			if len(words) == 1:
				graph.add_node(words[0])
			elif len(words) == 2: # two words -- unweighed graph
				if directed:
					graph.add_arc([words[0], words[1]])
				else:
					graph.add_edge([words[0], words[1]])
			elif len(words) > 2: # more than two words - labeled graph
				graph.weighted = True
				if directed:
					graph.add_arc([words[0], words[1]])
					graph.graph[words[0]][-1] = (words[1], words[2]) # process label
				else:
					graph.add_edge([words[0], words[1]])
					graph.graph[words[0]][-1] = (words[1], words[2])
					graph.graph[words[1]][-1] = (words[0], words[2])
		file.close()
		return graph

	@staticmethod
	def random_graph(nodes_num: int, prob: float, weighed: bool = False):
		"""
		Generates a random graph provided a number of nodes and probability of generating an edge.
		"""
		rand_graph = Graph()
		for i in range(1, nodes_num + 1):
			rand_graph.add_node(i)
			for j in range(1, i):
				if random() < prob:
					weight = randint(1, 10)
					rand_graph.add_edge([i, j], weight)
		
		return rand_graph

	def __str__(self):
		res = ""
		for v in self.graph:
			res += f"{v}:"
			for u in self.graph[v]:
				res += f" {u}"
			res += "\n"
		return res

	def add_node(self, node):
		"""
		Adds a node to a graph.
		"""
		if node not in self.graph:
			self.graph[node] = []

	def del_node(self, node):
		"""
		Recursively removes a node from a graph.
		"""
		if node in self.graph:
			self.graph.pop(node)
			for key in [*self.graph.keys()]:
				if node in self.graph[key]:
					self.graph[key].remove(node)

	def add_arc(self, arc: list):
		"""
		Adds arc to a graph provided a list of nodes.
		"""
		u, v = arc
		self.add_node(u)
		self.add_node(v)
		if v not in self.graph[u]:
			self.graph[u].append(v)

	def add_edge(self, edge: list, weight: int):
		"""
		Adds edge to a graph provided a list of nodes.
		"""
		u, v = edge
		if u == v:
			raise ValueError("Pętla!")
		self.add_node(u)
		self.add_node(v)
		if v not in self.graph[u]:
			self.graph[u].append((v, weight))
		if u not in self.graph[v]:
			self.graph[v].append((u, weight))

	def _array_to_dict(arr: np.array, nodes: list) -> dict:
		"""
		Converts a graph in array form to a graph in dict form.
		"""
		res_dict = {}
		for i, node in enumerate(nodes):
			neighbours = [nodes[j] for j, edge in enumerate(arr[i]) if edge]
			res_dict[node] = neighbours
		return res_dict

	def _dict_to_array(self, _dict: dict) -> np.array:
		"""
		Converts a graph in dict form to a graph in array form.
		"""
		n = len(_dict)
		nodes = [*_dict.keys()]
		res_arr = np.zeros(shape = (n, n), dtype=int)
		for u,v in [
			(nodes.index(u), nodes.index(v))
			for u, row in _dict.items() for v in row
		]:
			res_arr[u][v] += 1
		return res_arr

def MinSpanningTree(graph: Graph):
	"""
	Algorytm Jarnika-Prima -- minimalne drzewa spinające
	Dla nieskierowanych grafów ważonych (wagi to liczby całkowite)
	Zwraca parę (waga, drzewo), gdzie waga to łączna waga drzewa
	a drzewo to minimalne drzewo spinające w formie grafu ważonego
	"""
	if not graph.weighted: # jak graf nie jest ważony - zwróć nic
		return None, None
	for v in graph.graph: # wybieram jakiś wierzchołek  grafu
		break
	tree = {v:[]}       # zalążek drzewa
	weight = 0          # łączna waga
	q = PriorityQueue() # pusta kolejka priorytetowa
	for (u, w) in graph.graph[v]:
		q.put((int(w), v, u))
	while not q.empty():
		(w, v, u) = q.get()
		if u not in tree:
			weight += w
			tree[u] = [(v, w)]
			tree[v].append((u, w))
			for (x, w) in graph.graph[u]:
				if not x in tree:
					q.put((int(w), u, x))
	if len(tree) < len(graph.graph):
		print("Graf niespójny - zwrócone drzewo dla jednej składowej")
	wtree = Graph(tree)
	wtree.weighted = True
	return wtree

def preorder(graph: Graph, node):
	def DFS(u):
		for v in graph.graph[u[0]]:
			if v[0] not in visited:
				visited.append(v[0])
				DFS(v)
	visited = [node]
	for child in graph.graph[node]:
		if child[0] not in visited:
			visited.append(child[0])
			DFS(child)
	return visited
