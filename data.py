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


def generate_graph(type="", size=10, weight="uniform", n=100, p=0.05, minweight=1, maxweight=10):
    """
    :param type: The types of synthetic graphs. SBM and HSBM for major evaluation, ER,Bi-partite,path,cycle available.
    :param size: No. of graphs to test.
    :param size: uniform = U@R sample between min and max weight; unweighted = all weights are 1.
    """
    graph_list = []
    if type=="sbm":
        ns = [20,30,20,30,50]
        k = len(ns)
        intra_prob = 0.7
        inter_prob = 0.1
        probs = [[0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    probs[i][j] = intra_prob
                else:
                    probs[i][j] = inter_prob
                    
        for i in range(size):
            G = nx.stochastic_block_model(ns,probs)
            if weight=="unweighted":
                graph_list.append(G)
            elif weight=="uniform":
                G0 = nx.Graph()
                for (u,v) in G.edges:
                    G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
                graph_list.append(G)

    elif type=="hsbm":
        ns = [20,30,20,30,50]
        k = len(ns)
        intra_prob = 0.3
        inter_prob = 0.1
        prob_matrix = get_prob_matrix_for_HSBM(inter_prob, intra_prob)
        for i in range(size):
            G = nx.stochastic_block_model(ns,prob_matrix)
            if weight=="unweighted":
                graph_list.append(G)
            elif weight=="uniform":
                G0 = nx.Graph()
                for (u,v) in G.edges:
                    G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
                graph_list.append(G)
    
    elif type=="path":
        G = nx.path_graph(n)
        if weight=="unweighted":
            graph_list.append(G)
        elif weight=="uniform":
            G0 = nx.Graph()
            for (u,v) in G.edges:
                G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
            graph_list.append(G0)

    elif type=="cycle":
        G = nx.cycle_graph(n)
        if weight=="unweighted":
            graph_list.append(G)
        elif weight=="uniform":
            G0 = nx.Graph()
            for (u,v) in G.edges:
                G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
            graph_list.append(G0)

    elif type=="er":
        for i in range(size):
            G = nx.erdos_renyi_graph(n,p)
            if weight=="unweighted":
                graph_list.append(G)
            elif weight=="uniform":
                G0 = nx.Graph()
                for (u,v) in G.edges:
                    G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
                graph_list.append(G0)

    # elif type=="bipartite":
    #     for i in range(size):
    #         G = bipartite.random_graph(bpleft, n-bpleft, p, seed=1001)
    #         if weight=="unweighted":
    #             graph_list.append(G)
    #         elif weight=="uniform":
    #             G0 = nx.Graph()
    #             for (u,v) in G.edges:
    #                 G0.add_edge(u,v, weight=np.random.rand()*(maxweight-1)+minweight)
    #             graph_list.append(G0)

    return graph_list

def get_prob_matrix_for_HSBM(p, q):
    prob_matrix = [[p, 3 * q, 2 * q, q, q],
                   [3 * q, p, 2 * q, q, q],
                   [2 * q, 2 * q, p, q, q],
                   [q, q, q, p, 2 * q],
                   [q, q, q, 2 * q, p]]

    return prob_matrix

def generate_dataset_graph(datatype):
    """
    This method generates the similarity graph G according to each datatype and parameter gamma.

    :param datatype: The datatype considered
    :param gamma: A parameter to control the similarity weights in the resulting graph
    :return: A networkx similarity graph G corresponding to each datatype
    """

    # The datasets considered
    data_tuples = [('IRIS', 3), ('WINE', 5), ('CANCER', 5), ('BOSTON', 5), ('NEWSGROUP', 2)]
    
    if datatype == 'IRIS':
        dataset = datasets.load_iris()
        data = dataset.data
        gamma = 5
        G = get_kernel_sim_graph_from_data(data, gamma)

    elif datatype == 'BOSTON':
        dataset = datasets.load_boston()
        data = dataset.data
        gamma = 0.65
        G = get_kernel_sim_graph_from_data(data, gamma)

    elif datatype == 'WINE':
        dataset = datasets.load_wine()
        data = dataset.data
        gamma = 0.65
        G = get_kernel_sim_graph_from_data(data, gamma)

    elif datatype == 'CANCER':
        dataset = datasets.load_breast_cancer()
        data = dataset.data
        gamma = 0.65
        G = get_kernel_sim_graph_from_data(data, gamma)

    elif datatype == 'NEWSGROUP':
        cats = [
                # 'alt.atheism',
                'comp.graphics',
                'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware',
                'comp.sys.mac.hardware',
                # 'comp.windows.x',
                # 'misc.forsale',
                # 'rec.autos',
                # 'rec.motorcycles',
                'rec.sport.baseball',
                'rec.sport.hockey',
                # 'sci.crypt',
                # 'sci.electronics',
                # 'sci.med',
                # 'sci.space',
                # 'soc.religion.christian',
                # 'talk.politics.guns',
                # 'talk.politics.mideast',
                # 'talk.politics.misc',
                # 'talk.religion.misc'
                ]

        dataset = datasets.fetch_20newsgroups(subset='train', remove={'headers', 'footers', 'quotes'}, categories=cats)
        vectorizer = TfidfVectorizer()
        data_sparse = vectorizer.fit_transform(dataset.data)
        data = data_sparse.todense()
        gamma = 0.00003

        G = get_kernel_sim_graph_from_data(data, gamma)

    else:
        raise Exception("Data type unknown.")

    return G

def get_kernel_sim_graph_from_data(data, gamma, scaled=True):
    """
    This method creates a similarity graph based on the provided data and the parameter gamma. Concretely we construct
    a similarity graph according to the sklearn implementation of rbf_kernel, where every datapoint is a vertex and
    every pair of distinct data points (x, y) are corrected with an edge of weight w_(xy) = exp(- gamma * ||x - y||^2).
    Notice that this is also the known gaussian kernel exp(- ||x - y||^2/ (2*sigma^2)), for gamma = 1/(2*sigma^2).

    :param data: The data to be converted to a similarity graph
    :param gamma: A parameter to control the similarity weights between data points
    :param scaled: A boolean value to determine whether the data should be scaled or not
    :return: A similarity networkx graph generated according to the rbf kernel
    """

    # Initialise the graph
    G = nx.Graph()

    # Scale the data if required
    if scaled is True:
        data = scale(data)

    # Construct the adjacency matrix using the rbf_kernel
    adj_matrix = rbf_kernel(data, data, gamma)

    # Set a threshold and add an edge (u, v) in G only if w_(uv) >= threshold
    threshold = 10 ** (-10)

    # Update the adjacency matrix according to the threshold value
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if i == j:
                adj_matrix[i][j] = 0.0
            elif adj_matrix[i][j] < threshold:
                adj_matrix[i][j] = 0.0

    # Construct graph from adj_matrix
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if adj_matrix[i][j] > 0:
                G.add_edge(i, j, weight=adj_matrix[i][j]*1)

    return G
