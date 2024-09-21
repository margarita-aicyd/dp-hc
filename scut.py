import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import queue
import pulp
import pylab
import itertools
import time
import sys
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg

# from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from math import ceil,inf,log2,factorial,exp
from networkx.algorithms import community, bipartite
from networkx.algorithms.traversal.depth_first_search import dfs_tree

########### Sparsest Cut ###############

def find_max_flow(G):
    model = pulp.LpProblem("Max flow", pulp.LpMinimize)
    # Linear Programming for finding max flow
    nodes = G.nodes
    l_keys = list(G.edges)
    for i in nodes:
        for j in nodes:
            if i < j and not (i,j) in G.edges:
                l_keys.append((i,j))
    lvars = pulp.LpVariable.dicts("dist",
        #((i, j) for i in nodes for j in nodes if i < j),
        l_keys,
        lowBound=0,
        cat='Continuous')
    # Objective Function
    model += (
        pulp.lpSum([lvars[e] for e in G.edges])
    )
    # Constraints
    model += pulp.lpSum(lvars) >= 1
    for i in nodes:
        for j in nodes:
            for k in nodes:
                if (i, j) in lvars and (i, k) in lvars and (k, j) in lvars:
                    model += pulp.lpSum([lvars[(i, j)] - lvars[(i, k)] - lvars[(k, j)]]) <= 0
                if (i, j) in lvars and (i, k) in lvars and (j, k) in lvars:
                    model += pulp.lpSum([lvars[(i, j)] - lvars[(i, k)] - lvars[(j, k)]]) <= 0
                if (i, j) in lvars and (k, i) in lvars and (k, j) in lvars:
                    model += pulp.lpSum([lvars[(i, j)] - lvars[(k, i)] - lvars[(k, j)]]) <= 0
    # Solve problem
    model.solve()
    pulp.LpStatus[model.status]
    d = {}
    for var in lvars:
        d[var] = lvars[var].varValue
    W = pulp.value(model.objective)
    return W, d

class LeightonRaoMinCut:
    def __init__(self, G):
        self.G = G
        self.W, self.d = find_max_flow(G)
        self.C = len(G.edges)
        self.c = 1
        self.n = len(G)
        self.delta = 1 / (2 * self.n**2)
        dist = {}
        for e in G.edges:
            dist[e] = {'dist' : self.d[e]}
        nx.set_edge_attributes(self.G, dist)

    def find_cut(self):
        print("============================================================")
        print("Input Graph:")
        print("W = f = {}".format(self.W))
        self.partition()
        self.divide_in_two()
        print("First part: ", self.first)
        print("Second part: ", self.second)
        print("Coloured by cut graph:")
        print("Ratio: ", nx.cut_size(self.G, self.first) / min(len(self.first), len(self.second)))
        plt.figure(figsize=(6, 6))
        pos=nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.first, node_color='r')
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.second, node_color='b')
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        plt.axis('off')
        plt.show()
        print("============================================================")

    def partition(self):
        self.parts = []
        #print("delta ", self.delta)
        #print("magick number ", 4 * self.W * np.log(self.n) / self.C)
        if self.delta <= 4 * self.W * np.log(self.n) / self.C:
            #print("First case...")
            for n in self.G:
                self.parts.append([n])
        else:
            #print("Second case...")
            G1 = nx.Graph()
            for e in self.G.edges:
                e_num = ceil(self.C * self.d[e] / self.W)
                if e_num == 0:
                    G1.add_edge(e[0], e[1], weight=0)
                else:
                    if e_num == 1:
                        G1.add_edge(e[0], e[1],weight=1)
                    else:
                        G1.add_edge(e[0], "{}_{}_{}".format(e[0], e[1], 0), weight=1)
                        for i in range(e_num - 1):
                            G1.add_edge("{}_{}_{}".format(e[0], e[1], i), \
                                        "{}_{}_{}".format(e[0], e[1], i + 1), weight=1)
                            G1.add_edge("{}_{}_{}".format(e[0], e[1], e_num - 1), e[1], weight=1)
            # print("G+ graph")
            # plt.figure(figsize=(12, 12))
            # nx.draw(G1, with_labels=True)
            # plt.show()
            self.G1 = G1.copy()
            C0 = 2 * self.C / self.n
            eps = self.W * np.log(self.n) / (self.delta * self.C)
            while True:
                v = -1
                for n in self.G:
                    if n in G1:
                        v=n
                if v == -1:
                    break
                dist = {}
                self.bfs(G1, v, dist)
                #print("dist {}, node {}".format(dist, v))
                C_prev = C0
                v_prev = [v]
                for i in range(1, len(G1.edges) + 1):
                    v_new = [j for j in G1 if j in dist and dist[j] <= i]
                    #print("v_new {}".format(v_new))
                    Ci = len(G1.subgraph(v_new).edges)
                    if (Ci < (1 + eps) * C_prev):
                        self.parts.append([i for i in v_prev if i in self.G])
                        #print("Ci {}, (1+e)C_prev {}".format(Ci, (1 + eps) * C_prev))
                        #print("v_prev {} {}".format(v_prev, i))
                        G1.remove_nodes_from(v_prev)
                        break
                    C_prev = Ci
                    v_prev = v_new

    def divide_in_two(self):
        for part in self.parts:
            if len(part) >= 2 * self.n / 3:
                #print("Found large (more then 2/3) component...")
                self.divide_by_T(part)
                return
        #print("All components have less then 1/3 of nodes...")
        self.parts.sort(key=self.sort_key)
        self.first = []
        self.second = []
        while self.parts:
            if len(self.first) < self.n / 3 and len(self.first) + len(self.parts[-1]) < 2 * self.n / 3:
                self.first += self.parts.pop(-1)
            else:
                self.second += self.parts.pop(-1)

    def divide_by_T(self, T):
        dist = {}
        self.bfs(self.G1, None, dist, T)
        R = inf
        v_prev = T
        v_len_prev = len(T)
        for i in range(1, len(self.G1.edges) + 1):
            v_new = [j for j in self.G1 if j in dist and j in self.G and dist[j] <= i]
            v_len = len(v_new)
            if len(self.G) == v_len_prev:
                break
            Ri = (v_len - v_len_prev) / (v_len_prev * (len(self.G) - v_len_prev))
            R = min(R, Ri)
            if Ri == R:
                self.first = v_prev
            v_len_prev = v_len
            v_prev = v_new
        self.second = [i for i in self.G if i not in self.first]

    def sort_key(self, list_):
        return len(list_)

    def bfs(self, G, s, dist, T = None):
        q = queue.Queue()
        if s is not None:
            q.put(s)
            dist[s] = 0
        visit = {}
        for v in G:
            visit[v] = False
        if T is not None:
            for v in T:
                dist[v] = 0
                q.put(v)
                visit[v] = True
        while not q.empty():
            v = q.get()
            visit[v] = True
            for u in G[v]:
                if not visit[u]:
                    q.put(u)
                    dist[u] = dist[v] + G[v][u]['weight']


def cut(G, S1, S2):
    c = 0.0
    for v in S1:
        for u in S2:
            if G.has_edge(u,v) and 'weight' in G.edges[u, v]:
                c = c + G.edges[u, v]['weight']
            elif G.has_edge(u,v):
                c = c+1.0
    return c

def form_bipartition(p, G):
    if len(p) == 1:
        return (p[0], frozenset())
    i_min = 0
    uni = frozenset(G.nodes)
    n = len(G.nodes)
    min_cut = cut(G, p[0], uni - p[0]) / (len(p[0]) * (n - len(p[0])))
    for i in range(1, len(p)):
        the_cut = cut(G, p[i], uni - p[i]) / (len(p[i]) * (n - len(p[i])))
        if the_cut < min_cut:
            min_cut = the_cut
            i_min = i
    return (p[i_min], uni - p[i_min])


def random_partition(G):
    part0 = np.random.choice(G.nodes, len(G.nodes) // 2, replace=False)
    part1 = np.setdiff1d(G.nodes, part0)
    return (part0, part1)
    
def sweep_set(A, v_2, degrees):
    """
    Given the adjacency matrix of a graph, and the second eigenvalue of the laplacian matrix, use the sweep set
    algorithm to find a sparse cut.

    :param A: The adjacency matrix of the graph to use.
    :param v_2: The second eigenvector of the laplacian matrix of the graph
    :param degrees: a list with the degrees of each vertex in the graph
    :return: The set of vertices corresponding to the optimal cut
    """

    # Calculate n here once
    n = A.shape[0]
    # Keep track of the best cut so far
    best_cut_index = None
    best_conductance = None

    # Keep track of the size of the set and the cut weight to make computing the conductance
    # straightforward
    total_volume = np.sum(degrees)
    set_volume = 0
    set_size = 0
    cut_weight = 0

    # Normalise v_2 with the degrees of each vertex
    D = sp.sparse.diags(degrees, 0)
    v_2 = D.power(-(1/2)).dot(v_2)

    # First, sort the vertices based on their value in the second eigenvector
    sorted_vertices = [i for i, v in sorted(enumerate(v_2), key=(lambda x: x[1]))]

    # Keep track of which edges to add/subtract from the cut each time
    x = np.ones(n)

    # Loop through the vertices in the graph
    for (i, v) in enumerate(sorted_vertices[:-1]):
        # Update the set size and cut weight
        set_volume += degrees[v]
        set_size += 1

        # From now on, edges to this vertex will be removed from the cut at each iteration.
        x[v] = -1

        additional_weight = A[v, :].dot(x)
        cut_weight += additional_weight

        if cut_weight < 0:
            raise Exception('Something went wrong in sweep set: conducatance negative!')
        # Calculate the conductance
        if min(set_volume, total_volume - set_volume) == 0:
            this_conductance = 1
        else:
            this_conductance = cut_weight / min(set_volume, total_volume - set_volume)

        # Check whether this conductance is the best
        if best_conductance is None or this_conductance < best_conductance:
            best_cut_index = i
            best_conductance = this_conductance

    # Return the best cut
    return sorted_vertices[:best_cut_index+1]


def cheeger_cut(G):
    """
    Given a networkx graph G, find the cheeger cut.

    :param G: The graph on which to operate
    :return: A set containing the vertices on one side of the cheeger cut
    """
    if G.number_of_nodes() == 0:
        raise Exception(f'Cheeger cut: Graph should not be empty!')

    if nx.is_connected(G) is False:
        return list(list(nx.connected_components(G))[0])

    if G.number_of_nodes() == 1:
        return list(G.nodes())

    if G.number_of_nodes() == 2:
        return [list(G.nodes)[0]]

    # Compute the key graph matrices
    adjacency_matrix = nx.adjacency_matrix(G, weight='weight')
    laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight')
    graph_degrees = [t[1] for t in nx.degree(G, weight='weight')]

    # Compute the second smallest eigenvalue of the laplacian matrix
    eig_vals, eig_vecs = sp.sparse.linalg.eigsh(laplacian_matrix, which="SM", k=2)
    v_2 = eig_vecs[:, 1]

    # Perform the sweep set operation to find the sparsest cut
    S = sweep_set(adjacency_matrix, v_2, graph_degrees)
    nodes = list(G.nodes())
    return [nodes[i] for i in S]
    
def topDownCluster_rec(G, splitter, offset):
    n = len(G.nodes)
    assert n > 1
    merges = [[0, 0, 0, 0] for i in range(0, n-1)]
    from queue import Queue
    bfs = Queue()
    bfs.put((G.nodes, -1, -1))
    count = n-2
    while not bfs.empty():
        vert, pusher, idx = bfs.get()
        if len(vert) == 1:
            merges[pusher][idx] = list(vert)[0]
            continue
        merges[count][3] = len(vert)
        if pusher >= 0:
            merges[pusher][idx] = n+count+offset
        H = G.subgraph(vert)
        if nx.is_k_edge_connected(H, 1):
            p = splitter(H)
        else:
            ps = list(nx.k_edge_subgraphs(H, 1))
            p = form_bipartition(ps, H)
        if len(p[0]) == 0 or len(p[1]) == 0:
            part0 = np.random.choice(H.nodes, len(H.nodes) // 2, replace=False)
            part1 = np.setdiff1d(H.nodes, part0)
            part0 = frozenset(part0)
            part1 = frozenset(part1)
        else:
            part0, part1 = p
        assert len(part0) >= 1 and len(part1) >= 1
        bfs.put((part0, count, 0))
        bfs.put((part1, count, 1))
        count -= 1
    assert count == -1
    assert(len(merges) == n-1)
    assert(len(np.unique(np.array(merges)[:, 0:2])) == 2*n - 2)
    return merges
