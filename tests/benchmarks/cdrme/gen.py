#-================================== dataset.py

from matplotlib import pyplot as plt
import scipy.io as io
import networkx as nx
from cdrme.utils import label_to_genome_converter

class Dataset():
    def __init__(self,G, genome_true_label, true_label, C):
         self.G = G
         self.genome_true_label = genome_true_label
         self.true_label = true_label
         self.C = C

class DatasetLoader():
    def _calc_comm(matrix_adj, matrix_label):
        num = 0
        comm_num = {}
        comm_color = {} 
        comm_color_1 = {} 
        colors = [k for k in range(500)]

        for comm in matrix_label[0]:
            for node in comm[0]:
                comm_num[node - 1] = num
                comm_color[node - 1] = colors[num]
                comm_color_1[node - 1] = colors[num]

            num += 1
        
        return dict({
            'G':nx.from_numpy_array(matrix_adj),
            'result':comm_num})
 
    def _load(data):
        dico = {}
        dataset = data
        true_labels = [0] * len(dataset['result'].items())
        for k,v in dataset['result'].items():
            dico[v] = dico[v] + [k] if dico.get(v) else [k]
            true_labels[k]=v

        return Dataset(
            dataset['G'],
            label_to_genome_converter(true_labels),
            true_labels,
            len([set(val) for val in dico.values()])
            )

    def from_graph(G: nx.Graph) -> Dataset:
        """
        Build a Dataset directly from a NetworkX graph G.
        If `communities` is provided, it should be a list of lists of nodes;
        otherwise, this method will look for a node‐attribute 'community'
        on each node and group them automatically.
        """
        n = G.number_of_nodes()
        true_labels = [None] * n

        # convert to your genome‐encoding
        genome_true = label_to_genome_converter(true_labels)

        return Dataset(
            G,                # your graph
            genome_true,      # genome‐encoded labels
            None,      # flat list of length n
            None       # list-of-lists of nodes
        )

    def lfr(n = 100,mu=0.1,tau1 = 2,tau2 = 1.5,
        average_degree=10,
        seed=10,
        max_degree=50,
        min_community=10,
        max_community=50,
        draw=False):
        
        from random import randint
        GG = nx.LFR_benchmark_graph(
            n, tau1, tau2, mu, 
            average_degree=average_degree,
            seed=seed,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
        )

        communities = [list(GG.nodes[v]["community"]) for v in GG]
        colors = []
        n = len(communities)
        for i in range(n):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        l = len(GG.nodes)
        comColor = [0] * l
        true_labels = [0] * l
        k = 0
        for comm in communities:
            for node in comm:
                true_labels[node] = k
                comColor[node] = colors[k]
            k+=1    

        if draw:
            fig, axs = plt.subplots(nrows=1 ,ncols=1, figsize=(10, 10))
            layout = nx.kamada_kawai_layout(GG)
            nx.draw_networkx(GG, 
                        node_color=comColor,
                        node_size=580, with_labels=True,
                                        font_size=13, font_color='black', pos=layout)
            
        return Dataset(
            GG,
            label_to_genome_converter(true_labels),
            true_labels,
            communities
            )

################################# gen.py

import sys  
from sklearn.metrics import normalized_mutual_info_score
import scipy.io as io
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import walker 
import numpy as np
from math import ceil
import random
from random import choices, randint, randrange, random
from typing import List, Optional, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
from heapq import nsmallest
from cdrme.utils import timer
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100  

Genome = List[Tuple[int,float]]
Population = List[Genome]
PopulateFunc = Callable[[nx.Graph,int], Population]
FitnessFunc = Callable[[Genome,nx.Graph], float]
SelectionFunc = Callable[[Population, FitnessFunc,list,float], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome,float], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome,nx.Graph, int,float], Genome]
PrinterFunc = Callable[[Population, int, FitnessFunc], None]
PlotFunc = Callable[[None], None]

def average_fitness(population:Population,fitness_func: FitnessFunc)->float:
    return sum([fitness_func(genome) for genome in population])/len(population)
    
def selection_pair(population: Population, fitness_func: FitnessFunc,fitnesses=None) -> Population:    
    return choices(
        population=population,
        weights= fitnesses,
        k=2)
    genome = choosen_genome

    hubs = {i: np.where(population[i][:,1] == 1)[0] for i in range(len(population))}

    index = np.array(sorted(list(zip(hubs.keys(),genome[list(hubs.values())])),key=lambda x:x[1][1]))[:,0]

    return [population[index[np.random.choice([0,1])]] , choosen_genome]

    return choices(
        population=population,
        weights=np.arange(0,len(population)),
        k=2)

def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    pass

def mutation(genome: Genome, graph:nx.Graph, num: int = 1, probability: float = 0) -> Genome:
    return genome
    hub = max(genome,key=lambda x:x[1])
    while num>0:
        num-=1
        i = randint(0,len(genome)-1)
        if genome[i][1] < probability:
            genome[i] = hub[0],genome[i][1]

    return genome

def print_stats(population: Population, generation_id: int, fitness_func: FitnessFunc):
    pass

def run_evolution(
        populate_func: PopulateFunc,
        fitness_func: FitnessFunc,
        fitness_limit: float,
        selection_func: SelectionFunc = selection_pair,
        crossover_func: CrossoverFunc = single_point_crossover,
        mutation_func: MutationFunc = mutation,
        generation_limit: int = 100,
        cross_mutation_ratio: float = 0.2,
        first_crossover_treshold=0.5,
        crossover_tresholds = None,
        do_print_timers = False,
        printer: Optional[PrinterFunc] = None) \
        -> Tuple[Population, int]:
    population = populate_func()

    def NormalizeData(data,reverse=False):
        if (np.max(data) - np.min(data))<=0.001:
            res = data/data
        else:
            res = (data - np.min(data)) / (np.max(data) - np.min(data))
        return 1-res if reverse else res
    new_generation = []
    old_parents = np.array(population,copy=True)
    with timer(do_print=do_print_timers,text="Pre Crossover"):
        for subset in itertools.combinations(population,2):
            parent_a = subset[0]
            parent_b = subset[1]
            a = crossover_func(parent_a,parent_b,threshold=first_crossover_treshold)
            new_generation+=[a]

    new_generation = np.array(new_generation)
    average_sim = NormalizeData(np.array(list(map(np.var,new_generation[:,:,0]))))
    average_fitness = NormalizeData(np.array([fitness_func(gen) for gen in new_generation]),True)
    population = new_generation[average_sim+average_fitness>=np.average(average_sim+average_fitness)]
    i = 0
    for i in range(generation_limit):
        with timer(do_print=do_print_timers,text=f"Calc Weights {i}"):

            if ceil(cross_mutation_ratio * generation_limit) <= i:
                population_fitness = np.array([ fitness_func(genome) for genome in population])
            else:
                population_fitness = np.array([ 1 for genome in population])
            population_fitness = ((1- population_fitness)+5)**2
        if printer is not None:
            printer(population, i, fitness_func)

        next_generation = []

        counter = min(2,len(population))        
        next_generation+=list(nsmallest(counter, population, key=lambda x: fitness_func(x)))
        with timer(do_print=do_print_timers,text="-----------------"):
            for j in range(len(population)-counter):
                with timer(do_print=do_print_timers,text="\tSelection"):
                    parents = selection_pair(population, fitness_func,population_fitness)
                with timer(do_print=do_print_timers,text="\tCrossover"):
                    if crossover_tresholds is None:
                        offspring_a  = crossover_func(parents[0], parents[1])
                    else :
                        
                        thresh = np.average(parents[1][:,1]) + np.average(parents[0][:,1])/2.0
                        offspring_a  = crossover_func(parents[0], parents[1],threshold=thresh)
                next_generation += [offspring_a]
        
        population = np.array(next_generation)

        # Filtering
        generation_fitness = np.array([fitness_func(gen) for gen in population])
        average_sim = np.array(list(map(np.average,population[:,:,0])))

        for i in range(len(generation_fitness)):
            if average_sim[i] >= np.average(average_sim):
                population[i]=mutation_func(population[i],probability=np.average(average_sim))
    return population, i

# ### Objective Function (Kernel KMeans - Ratio Cut)
def kkm_rc(G:nx.Graph,individual):
    kkm = 0
    rc = 0
    # all_coms [c1,c1,c1,c2]
    all_comms = [c for c,p in individual]
    
    # each comm inner is set to 0
    # inner_comm & outter_comm [c1:0,c1:0,c1:0,c2:0]
    inner_comm = {item:0 for item in set(all_comms)}
    outter_comm = {item:0 for item in set(all_comms)}

    # initialize comm size 
    comm_size = {}
    for node in all_comms:
        comm_size[node] = comm_size[node]+1 if comm_size.get(node) else 1

    # TODO this could be faster if we dont check node and neighbour twice ( now we check v -> neigh then we do neigh -> v too)
    # fill comm inner
    for node_index in range(len(all_comms)):
        neighbours = G.neighbors(node_index)
        for neigh in neighbours:
            if all_comms[node_index] == all_comms[neigh]:
                inner_comm [all_comms[node_index]]+=1
            else:
                outter_comm[all_comms[node_index]]+=1

    # calc fitness (kkm,rc)
    # calc kkm
    inner_sum = max(sum(list(inner_comm.values())),1)
    for comm,inner in inner_comm.items():
        kkm+= (inner/2/inner_sum)
        # ((comm_size[comm]*(comm_size[comm]-1))+1))
    # calc rc
    outer_sum = max(sum(list(outter_comm.values())),1)
    for comm,outter in outter_comm.items():
        rc+= ((outter)/2/outer_sum)
    # print(f"inner-> {inner_sum} outer-> {outer_sum}")
    kkm = 1 - kkm
    return kkm,rc

def kkm_rc_fitness(genome, graph: nx.Graph) -> float:
    return sum(kkm_rc(graph,genome))
    # TODO: normalize inner outer links based on graph density or total links

from random import choice, randint


def random_walks(G, n_walks: int, walk_len: int, start_nodes: list):
    """
    Perform simple uniform random walks on a networkx.Graph.
    G: networkx.Graph
    n_walks: total number of walks per start node
    walk_len: length of each walk
    start_nodes: list of seed nodes
    Returns: list of walks (each a list of node IDs)
    """
    walks = []
    for src in start_nodes:
        for _ in range(n_walks):
            walk = [src]
            current = src
            for _ in range(walk_len - 1):
                nbrs = list(G.neighbors(current))
                if not nbrs:
                    break
                current = choice(nbrs)
                walk.append(current)
            walks.append(walk)
    return walks

# ## `Population Initialization`
def create_walks(G:nx.Graph,n_wlaks,walk_len,start_nodes,percentage):
    walks = random_walks(G,n_walks=n_wlaks,walk_len=walk_len,start_nodes=start_nodes)
    # TODO: weighted random walk based on similarity (number of neighbours) based on hub or walk node or both
    rw_node_occurence = {}
    for walk in walks:
        for node in walk: 
            rw_node_occurence[node] = rw_node_occurence[node] + 1 if rw_node_occurence.get(node) else 1

    arr =np.array(sorted(rw_node_occurence.items(),key= lambda x:x[1],reverse=True))
    # print(arr[:int(percentage*0.01*arr[0][1])])
    return arr[:int(percentage*0.01*arr[0][1])]

def sorted_hub_nodes(graph: nx.Graph,selection_type = 0):
    dico =graph.degree()
    from math import ceil
    average_degree = np.average(np.array(list(graph.degree))[:,1])
    max_degree = np.max(np.array(list(graph.degree))[:,1])
    
    if selection_type == 0:
        result = np.array(sorted (list(graph.degree),key= lambda x:abs(average_degree - x[1])))
        return result,np.array(1/(np.abs(average_degree - result[:,1])+1 ))
    
    elif selection_type ==1:
        result = np.array(sorted (list(graph.degree),key= lambda x:abs(max_degree - x[1])))
        return result,np.array(abs(max_degree - result[:,1]))

    elif selection_type ==2:
        degs = {}
        for n,d in dico:
            if degs.__contains__(d):
                degs[d].append(n)
            else:
                degs[d] = []
        result = np.array(sorted (list(graph.degree),key= lambda x:len(degs[x[1]])))
        return result,np.array([len(degs[x[1]]) for x in result])
# ## Generate Chromosome
from cdrme.simmilarity import Simmilarity
hub_lines = []
def generate_population(G,percentage,rw_percentage,n_wlaks,walk_len,remove_repeated_hubs=False,hub_selection_type=0) -> 'Population':
    global hub_lines
    hub_lines= []
    genomes = []
    selected = 0
    desiered_hub_population = ceil(percentage/100* len(G.nodes()))
    hubs,weight =  sorted_hub_nodes(G,selection_type=hub_selection_type)
    hubs_dict = dict(zip([n for n,d in hubs],weight))

    # print(hubs_dict)
    while len(hubs_dict)>0:
        if selected >= desiered_hub_population:
            break
        hubs = list(hubs_dict.keys())
        # print(hubs)
        hub = choices(hubs,weights=weight,k=1)[0]
        # hub = choices(hubs,weights=hubs[:,1],k=1)[0][0][0]
        # print(hub)
        selected+=1
        rw_nodes = create_walks(G,n_wlaks=n_wlaks,walk_len=walk_len,start_nodes=[hub],percentage=rw_percentage)
        hub_lines.append(rw_nodes)
        for node,freq in rw_nodes:
            # hubs_dict[node][1]/=(50*freq/max(rw_nodes,key=lambda x:x[1])[1])
            hubs_dict[node] = hubs_dict[node]/freq

        genomes.append(generate_genome(hub, G,rw_nodes))    
    return genomes

def generate_genome(hub,G:nx.Graph,rw_nodes=None) -> 'Genome':
    return np.array([[hub,Simmilarity.get_simmilarity(G,node,hub,rw_nodes)] for node in G.nodes()])


from cdrme.utils import genome_to_label_converter

def NMI(genome:Genome) -> int:
    return normalized_mutual_info_score (dataset.true_label,genome_to_label_converter(genome))

def mutation(genome: Genome,graph: nx.Graph,majaroity_density:float=0.5, probability: float = 0) -> Genome:
    genome = np.array(genome,copy=True)
    for i in range(len(genome)):
        if genome[i][1] <= probability:
            choosen_neighbor = list(graph.neighbors(i))
            if len(choosen_neighbor):
                negh_comm_size = {}
                for negh in choosen_neighbor:
                    c = genome[negh][0]
                    #TODO Add Prof's Formula
                    negh_comm_size[c] = [negh_comm_size[c][0] + 1,negh_comm_size[c][1]+genome[negh][1]] \
                        if negh_comm_size.get(c) \
                            else [1,genome[negh][1]]
                most_populated_comm = max(negh_comm_size.items(),key=lambda x:x[1][1])
                if most_populated_comm[1][1]/most_populated_comm[1][0] < majaroity_density :
                    # genome[i] = i,1 if len(choosen_neighbor) else genome[i]
                    genome[i] =  genome[i]
                # else:
                genome[i] = most_populated_comm[0],most_populated_comm[1][1]/most_populated_comm[1][0] if len(choosen_neighbor) else genome[i]
                        
    return genome    

# ## `Crossover` 
def crossover(a: Genome, b: Genome, graph:nx.Graph,  threshold=.8,lower_bound = -0.1, do_print=False) -> Tuple[Genome, Genome]:
    redirect_hubs = {}

    def get_redirect(hub):
        temp = hub
        hub_dic = {hub:1}
        if redirect_hubs.get(hub) == None:
            return hub
        while(redirect_hubs[hub]!= hub ):
            
            if hub_dic[hub] >2:
                return min(list(filter(lambda x:x[1]>=2,hub_dic.items() )),key = lambda x:x[0])[0]

            hub = redirect_hubs[hub]
            if not hub_dic.get(hub):
                hub_dic[hub]=1
            else:
                hub_dic[hub]+=1
        
        return hub

    def get_redirect_similarity(hub, similarity):
        temp = hub
        sum_of_sims = 0
        dept = 1
        while(redirect_hubs[hub]!= hub and redirect_hubs[hub] != temp):
            try:
                sum_of_sims += similarity[redirect_hubs[hub]].get(hub)
            except:
                print("______")
                print(redirect_hubs)
                print("______")
                print(similarity,hub,redirect_hubs[hub])
            hub = redirect_hubs[hub]
            dept+=1
        return sum_of_sims/dept

    hubs_in_parent_a = a[a[:,1]==1]
    hubs_in_parent_b = b[b[:,1]==1]
	
    # """>>>{0.0: {2.0: 0.8}, 3.0: {4.0: 0.9, 5.0: 0.9}, 2.0: {0.0: 0.4}, 4.0: {3.0: 0.8}, 5.0: {}}"""
    hub_similarity = {}
   
    for hub in hubs_in_parent_a:
        redirect_hubs[hub[0]] = hub[0]
        for hub_in_b in hubs_in_parent_b:
            if not hub_similarity.get(hub[0]):
                hub_similarity[hub[0]] = {}
            if a[int(hub_in_b[0])][0] == hub[0] : #and a[int(hub_in_b[0])][1] :#>= threshold:
                hub_similarity[hub[0]][hub_in_b[0]] =  a[int(hub_in_b[0])][1]

    for hub in hubs_in_parent_b:
        redirect_hubs[hub[0]] = hub[0]
        for hub_in_a in hubs_in_parent_a:
            if not hub_similarity.get(hub[0]):
                hub_similarity[hub[0]] = {}
            if b[int(hub_in_a[0])][0] == hub[0] : #and b[int(hub_in_a[0])][1] >= threshold:
                hub_similarity[hub[0]][hub_in_a[0]] =  b[int(hub_in_a[0])][1]


    # print(hub_similarity)
    # """>>>{0.0: {2.0: 0.8}, 3.0: {4.0: 0.9, 5.0: 0.9}, 2.0: {0.0: 0.4}, 4.0: {3.0: 0.8}, 5.0: {}}"""

    for outer,hubs_connected_to_outer in hub_similarity.items():
        for hub,sim_val in hubs_connected_to_outer.items():
            if sim_val < threshold:
                continue
            hub_sim_to_outer =  hub_similarity[hub].get(outer)
            if hub_sim_to_outer :
                master_slave_sim = max(sorted([(outer,hub,sim_val),(hub,outer,hub_sim_to_outer)],key=lambda x:x[0]),key = lambda x:x[2])
                redirect_hubs[ master_slave_sim[1] ] = master_slave_sim[0]
            else:
                redirect_hubs[hub]=outer
    offspring= np.array(a,copy=True)

    visite = np.zeros(len(offspring))
    for node_index in range(len(offspring)):
        aa = a[node_index]
        bb = b[node_index]
        hub_in_a = aa[0]
        hub_in_b = bb[0]
        is_meged_a = hub_in_a != get_redirect(hub_in_a)
        is_meged_b = hub_in_b != get_redirect(hub_in_b)

        # print(f'a: {hub_in_a} - {is_meged_a} \t b: {hub_in_b} - {is_meged_b}')
        if node_index == get_redirect(hub_in_a):                                # if node index was hub continue
            continue
        if not is_meged_a and not is_meged_b :                                  # if nither a's or b's hub aren't merged
            offspring[node_index] = max(aa,bb,key=lambda x:x[1])
        elif get_redirect(hub_in_a) == hub_in_b \
            or get_redirect(hub_in_b) == hub_in_a:                              # if a's hub merged to b. Or vice versa                             
            offspring[node_index] = [get_redirect(hub_in_a),
                                    np.average([aa[1],bb[1]])]
        elif get_redirect(hub_in_a) == get_redirect(hub_in_b):                  # if a's and b's hub merged to the same hub
            a_half = get_redirect_similarity(hub_in_a,hub_similarity)
            b_half = get_redirect_similarity(hub_in_b,hub_similarity)

            if a_half == None:
                a_half = hub_similarity[hub_in_b].get(hub_in_a)

            if b_half == None:
                b_half = hub_similarity[hub_in_a].get(hub_in_b)
            
            alpha = threshold
            redirected_sim = ((1-alpha) * np.average([a_half,b_half]))  + (alpha * np.average([aa[1],bb[1]]))

            offspring[node_index] = [get_redirect(hub_in_a), redirected_sim]
            
        elif get_redirect(hub_in_a) != hub_in_a and get_redirect(hub_in_b) != hub_in_b:
            a_half = get_redirect_similarity(hub_in_a,hub_similarity)
            b_half = get_redirect_similarity(hub_in_b,hub_similarity)
            offspring[node_index] = [get_redirect(hub_in_a),np.average([aa[1],a_half])] \
                                        if np.average([aa[1],a_half]) > np.average([bb[1],b_half]) \
                                            else [get_redirect(hub_in_b),np.average([bb[1],b_half])]
        else :                                                                      # (get_redirect(hub_in_a) == hub_in_a) or (get_redirect(hub_in_b) != hub_in_b)
            offspring[node_index] = max([[get_redirect(hub_in_a),aa[1]],[get_redirect(hub_in_b),bb[1]]],key=lambda x:x[1])
    return offspring

from functools import partial

from cdrme.utils import genome_to_comm_convertor

def run_cdrme(G):
    dataset = DatasetLoader.from_graph(G)

    # @o\\:veira: I keeped the magic numbers, dont have idea why theyre here.
    mutation_ratio = 0
    g_limit = 10
    population_percentage = 10
    thresholds = np.linspace(start=0, stop=0, num=g_limit) ** 0.5

    population, generations = run_evolution(
        populate_func=partial(generate_population,
                              G=dataset.G,
                              percentage=population_percentage,
                              rw_percentage=10,
                              n_wlaks=500,
                              walk_len=4,
                              hub_selection_type=0,
                              remove_repeated_hubs=False),
        fitness_func=partial(kkm_rc_fitness, graph=dataset.G),
        cross_mutation_ratio=mutation_ratio * 0.1,
        fitness_limit=.001,
        generation_limit=g_limit,
        first_crossover_treshold=0.9,
        crossover_tresholds=thresholds,
        crossover_func=partial(crossover,
                               graph=dataset.G,
                               lower_bound=-0.01),
        mutation_func=partial(mutation,
                              graph=dataset.G,
                              probability=1.1),
        do_print_timers=False
    )

    # pick the single best individual
    best_gene = max(
        population,
        key=lambda x: nx.community.modularity(
            dataset.G,
            genome_to_comm_convertor(x).values()
        )
    )

    comm2nodes = genome_to_comm_convertor(best_gene)
    # comm2nodes is a dict: community_id → [node, node, …]
    # Convert to exactly the same type as louvain_communities (a list of sets):
    communities = [set(nodes) for nodes in comm2nodes.values()]
    return communities


    return comm2nodes