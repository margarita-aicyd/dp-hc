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

from scut import *

def getNoisedGraph(G, epsilon, positive=True):
    H = nx.Graph()
    H.add_nodes_from(G)
    n = len(G.nodes)
    for i,j in list(itertools.combinations(G.nodes, r=2)):
        if G.has_edge(i,j) and 'weight' in G.edges[i,j]:
            dij = G.edges[i,j]['weight']
        elif G.has_edge(i,j):
            dij = 1
        else:
            dij = 0
            
        if epsilon is not None:
            ### 1st step of perturbation ###
            dij += 5*(log2(n) / epsilon)
            ### 2nd step of perturbation ###
            dij += np.random.laplace(0, 1.0 / epsilon)
        if G.has_edge(i,j):
            if dij > 0:
                H.add_edge(i,j, weight=dij)
            if dij < 0:
                H.add_edge(i,j, weight=0)
            
    return H

def input_perturb(G, epsilon, positive=True):
    H = nx.Graph()
    H.add_nodes_from(G)
    n = len(G.nodes)
    for i,j in list(itertools.combinations(G.nodes, r=2)):
        if G.has_edge(i,j) and 'weight' in G.edges[i,j]:
            dij = G.edges[i,j]['weight']
        elif G.has_edge(i,j):
            dij = 1
        else:
            dij = 0
        if epsilon is not None:
            ### 1st step of perturbation ###
            dij += np.random.laplace(0, 1.0 / epsilon)
        if G.has_edge(i,j):
            if dij > 0:
                H.add_edge(i,j, weight=dij)
            if dij < 0:
                H.add_edge(i,j, weight=0)      
    # print("======================")
    # for i,j in list(itertools.combinations(H.nodes, r=2)):
    #     if H.has_edge(i,j):
    #         print(H.edges[i,j]['weight'])
    
    return H

def dphc(G, epsilon, model="weight", split_type='cheeger', print_time=False):
    if model=="ip":
        H = input_perturb(G,epsilon,True)
    elif model=="weight":
        H = getNoisedGraph(G, epsilon, True)
    elif model=="non_priv":
        H = G.copy()

    n = len(G.nodes)
    start = time.time()
    if split_type == 'greedy':
        def splitter(G):
            if G.size('weight') == 0:
                return random_partition(G)
            return form_bipartition(community.greedy_modularity_communities(G, weight='weight'),
                             G)
    elif split_type == 'leightonrao':
        def splitter(G):
            if G.size('weight') == 0:
                return random_partition(G)
            if len(G.nodes) <= 5:
                return form_bipartition(community.greedy_modularity_communities(G, weight='weight'),
                             G)
            sp = LeightonRaoMinCut(G)
            sp.partition()
            sp.divide_in_two()
            return (sp.first, sp.second)
    elif split_type == 'cheeger':
        def splitter(G):
            if G.size('weight') == 0:
                return random_partition(G)
            v1 = cheeger_cut(G)
            v2 = list(set(list(G.nodes()))-set(v1))
            return (v1,v2)
    else:
        raise Exception('%s not an available split option' % split_type)
    merges = topDownCluster_rec(H, splitter, 0)
    end = time.time()
    if print_time:
        print("Computation time of our ALG is: ", end-start)
    return merges, G
    
def cost(dendro, G0, return_levels=False):
    cluster_cuts = {}
    NUMBEROFELEMENTS = G0.number_of_nodes()
    for i in range(NUMBEROFELEMENTS):
        cluster_cuts[i]= {}

    for i, j in list(itertools.combinations(range(NUMBEROFELEMENTS), r=2)):
        cutij = cut(G0, [j], [i])
        cluster_cuts[i][j]= cluster_cuts[j][i] = cutij

    cost = 0
    costs = []
    current_clusters = [i for i in range(NUMBEROFELEMENTS)]
    for step in range(len(dendro)):
        new_clust_id = step + NUMBEROFELEMENTS
        costNODE = cluster_cuts[dendro[step][0]][dendro[step][1]] * dendro[step][3]
        current_clusters.remove(dendro[step][0])
        current_clusters.remove(dendro[step][1])
        v0 = int(dendro[step][0])
        v1 = int(dendro[step][1])
        if v0 < NUMBEROFELEMENTS:
            s0 = 1
        else:
            s0 = dendro[v0-NUMBEROFELEMENTS][3]
        if v1 < NUMBEROFELEMENTS:
            s1 = 1
        else:
            s1 = dendro[v1-NUMBEROFELEMENTS][3]
        cluster_cuts[new_clust_id] = {}
        for j in current_clusters:
            cluster_cuts[j][new_clust_id] = cluster_cuts[new_clust_id][j] = cluster_cuts[dendro[step][0]][j] + cluster_cuts[dendro[step][1]][j]
        current_clusters.append(new_clust_id)
        cost+=costNODE
        costs.append([costNODE, costNODE / (s0 * s1), s0 * s1])
    if return_levels:
        return (cost, costs)
    return cost

def run_alg(model,graph_list,epsilon_list, split="cheeger"):
    cost_per_graph = []
    for G in graph_list:
        cost_per_eps = []
        for eps in epsilon_list:
            priv_output, original_G = dphc(G, eps, model, split)
            priv_cost = cost(priv_output, original_G)
            cost_per_eps.append(priv_cost)
        cost_per_graph.append(cost_per_eps)

    return cost_per_graph
