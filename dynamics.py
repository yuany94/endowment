import numpy as np
import networkx as nx

W = np.loadtxt('output/log_karate_4_W_2.txt')
B = np.loadtxt('output/log_karate_4_B_2.txt')
C = np.loadtxt('output/log_karate_4_C_2.txt')
beta = np.loadtxt('output/log_karate_4_beta_2.txt')
f = open('data/karate.txt')

K = 4
midpoint = 2

line = f.readline()

G = nx.Graph()

while line:
    items = line.split(' ')
    G.add_edge(int(items[0]), int(items[1]))
    line = f.readline()

Hs = {}

for reduction in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:

    N = len(W)

    U = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            U[i, j] = np.dot(B, np.maximum(W[j][:midpoint]-W[i][:midpoint], 0.0)) - np.linalg.norm(np.multiply(reduction * C, W[j][midpoint:]-W[i][midpoint:]))

    V = np.minimum(U, U.T)

    ids = []
    utilities = []

    for i in range(N):
        for j in range(i+1, N):
            utilities.append(V[i, j])
            ids.append((i, j))

    
    if reduction == 1.0:
        arg_utilities = np.argsort(utilities)
        reverse_utility = utilities[arg_utilities[-2*len(G.edges())]]

    H = nx.Graph()
    H.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            if i < j and U[i, j] >= reverse_utility:
                H.add_edge(i, j)


    Hs[reduction] = H
    
for key in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    H = Hs[key]
    sp = []
    node_num = len(H.nodes())
    for _ in range(5000):
        i = H.nodes()[np.random.randint(node_num)]
        j = H.nodes()[np.random.randint(node_num)]
        if i != j:
            try:
                sp.append(nx.shortest_path_length(H, i, j))
            except:
                pass

    diversity = []
    edge_num = len(H.edges())
    pairs = np.random.randint(0, edge_num, 5000)
    edge_list = H.edges()
    for n in pairs:
        i, j = edge_list[n]
        if i != j:
            diversity.append(np.linalg.norm(np.multiply(C, W[i][midpoint:]-W[j][midpoint:])) / np.linalg.norm(C))
            
    base_diversity = []
    for _ in range(5000):
        i = H.nodes()[np.random.randint(node_num)]
        j = H.nodes()[np.random.randint(node_num)]
        base_diversity.append(np.linalg.norm(W[i][midpoint:]-W[j][midpoint:]))

    print(key, len(H), len(H.edges()), nx.density(H), nx.average_clustering(H), np.mean(sp), np.mean(diversity))