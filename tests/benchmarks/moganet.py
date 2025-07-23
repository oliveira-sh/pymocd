import random
import math
import networkx as nx

def detect_communities_ga(G,
                          pop_size=100,
                          generations=50,
                          crossover_rate=0.8,
                          mutation_rate=0.1,
                          r=1.0,
                          elite_ratio=0.1):
    """
    GA-Net: Genetic Algorithm for Community Detection.
    
    Parameters
    ----------
    G : networkx.Graph
        Undirected graph whose communities we wish to detect.
    pop_size : int
        Number of candidate partitions in each generation.
    generations : int
        Number of GA iterations.
    crossover_rate : float in [0,1]
        Probability of performing crossover on a selected pair of parents.
    mutation_rate : float in [0,1]
        Per-gene mutation probability.
    r : float
        Exponent in the power‐mean of the community score.
    elite_ratio : float in [0,1]
        Fraction of top individuals preserved each generation.
    
    Returns
    -------
    dict[node, community_id]
        Mapping from each node in G to an integer community label.
    """
    # 1) Remap nodes to indices 0..N-1 for internal representation
    nodes = list(G.nodes())
    N = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    inv_idx = {i: node for node, i in idx.items()}
    
    # 2) Precompute each node's neighbors (by index)
    neighbor_indices = [[] for _ in range(N)]
    for u, v in G.edges():
        i, j = idx[u], idx[v]
        neighbor_indices[i].append(j)
        neighbor_indices[j].append(i)
    
    # 3) Decode a genotype to communities via Union-Find
    def decode(ind):
        parent = list(range(N))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i, j in enumerate(ind):
            union(i, j)
        comps = {}
        for i in range(N):
            root = find(i)
            comps.setdefault(root, []).append(i)
        return list(comps.values())
    
    # 4) Compute the community-score fitness of an individual
    def fitness(ind):
        CS = 0.0
        for C in decode(ind):
            m = len(C)
            if m <= 1:
                continue
            # intra‐community degree counts
            d_intra = [ sum(1 for j in neighbor_indices[i] if j in C) for i in C ]
            # power‐mean M(S)
            M = sum((d / m) ** r for d in d_intra) / m
            # volume v_S = sum of intra‐degrees
            vS = sum(d_intra)
            CS += M * vS
        return CS
    
    # 5) Initialization: each gene links to a random neighbor
    def random_individual():
        ind = []
        for i in range(N):
            nbrs = neighbor_indices[i]
            ind.append(random.choice(nbrs) if nbrs else i)
        return ind
    
    population = [random_individual() for _ in range(pop_size)]
    fitnesses = [fitness(ind) for ind in population]
    
    # 6) Roulette-wheel parent selection
    def select_parent():
        total = sum(fitnesses)
        if total == 0:
            return random.choice(population)
        pick = random.random() * total
        cum = 0.0
        for ind, fit in zip(population, fitnesses):
            cum += fit
            if cum >= pick:
                return ind
        return population[-1]
    
    # 7) GA main loop
    elite_size = max(1, int(elite_ratio * pop_size))
    for gen in range(generations):
        # a) Keep the elites
        ranked = sorted(zip(population, fitnesses),
                        key=lambda x: x[1], reverse=True)
        new_pop = [ind for ind, _ in ranked[:elite_size]]
        
        # b) Fill the rest of the new population
        while len(new_pop) < pop_size:
            p1 = select_parent()
            p2 = select_parent()
            # crossover
            if random.random() < crossover_rate:
                # uniform crossover
                mask = [random.random() < 0.5 for _ in range(N)]
                c1 = [p1[i] if mask[i] else p2[i] for i in range(N)]
                c2 = [p2[i] if mask[i] else p1[i] for i in range(N)]
            else:
                c1, c2 = p1[:], p2[:]
            # mutation
            for child in (c1, c2):
                for i in range(N):
                    if random.random() < mutation_rate:
                        nbrs = neighbor_indices[i]
                        child[i] = random.choice(nbrs) if nbrs else i
                new_pop.append(child)
                if len(new_pop) == pop_size:
                    break
        
        # c) Evaluate new population
        population = new_pop
        fitnesses = [fitness(ind) for ind in population]
    
    # 8) Decode the best individual
    best = max(zip(population, fitnesses), key=lambda x: x[1])[0]
    communities = decode(best)
    
    # 9) Build final node→community mapping
    node2comm = {}
    for cid, comp in enumerate(communities):
        for i in comp:
            node2comm[inv_idx[i]] = cid
    return node2comm