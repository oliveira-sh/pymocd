# CDRME reference implementation, supplied privately by the authors as the
# Jupyter notebook gen.ipynb (Python 3.11.5). Converted to plain Python: the
# code cells verbatim and in order, markdown headings kept as section markers,
# cell outputs discarded. Nothing inside a code cell was altered except the
# shell escapes marked below, which are not valid Python.

# ## Imports

# !pip install pybind11

# !pip install graph-walker

# !pip install networkx

# !python --version 

"Tested Python version: 3.11.5";

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
from gen.dataset import Dataset,DatasetLoader
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

class DrawGeneticData():
    instance : list[Genome,Genome,Genome]= []

colors = []
def refresh_colors(G):
    global colors
    colors = []
    n = len(G.nodes())
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

# ### Load Dataset

dataset:Dataset = DatasetLoader.karate

# ### Methods

def fitness(genome: Genome, fitness_func: FitnessFunc) -> int:
    pass

def average_fitness(population:Population,fitness_func: FitnessFunc)->float:
    return sum([fitness_func(genome) for genome in population])/len(population)
    
def generate_population_basic(size: int, genome_length: int, C: int = 2) -> Population:
   pass

def selection_pair(population: Population, fitness_func: FitnessFunc,fitnesses=None) -> Population:    
    # choosen_genome = populationchoices(population)[0]
    # choosen_genome = choices(population,k=2) if len(population)>=2 else [population[0],population[0]] 
    # return choosen_genome
    
    #TODO: fix selection based on weights
    # population = sorted(
            # population, key=lambda genome: fitness_func(genome), reverse=True)
    # return choices(
    #     population=population,
    #     weights=np.arange(0,len(population)),
    #     k=2)

    return choices(
        population=population,
        weights= fitnesses,
        k=2)

    # return choices(
    #     population=population,
    #     k=2)

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

from heapq import nsmallest
from gen.utils import timer


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
    # for i in range(10):
    with timer(do_print=do_print_timers,text="Pre Crossover"):
        for subset in itertools.combinations(population,2):
            # while len(population):
            # parent_a = population.pop(randint(0,len(population)-1))
            # parent_b = population.pop(randint(0,len(population)-1)) if len(population) else parent_a[:]
            parent_a = subset[0]
            parent_b = subset[1]
            a = crossover_func(parent_a,parent_b,threshold=first_crossover_treshold)
            # a = crossover_func(parent_a,parent_b)
            new_generation+=[a]

        # population.extend(old_parents)
    new_generation = np.array(new_generation)
    average_sim = NormalizeData(np.array(list(map(np.var,new_generation[:,:,0]))))
    average_fitness = NormalizeData(np.array([fitness_func(gen) for gen in new_generation]),True)
    # print(f"f = {average_fitness}")
    # print(f"s = {average_sim}")
    # print(f"nmi = {[NMI(gen) for gen in new_generation]}")
    population = new_generation[average_sim+average_fitness>=np.average(average_sim+average_fitness)]
    # print(f"selected : {[fitness_func(gen) for gen in population]}")
    print(f"first generation pruning : {len(new_generation)} -> {len(population)}")
    # population =  new_generation
    i = 0
    for i in range(generation_limit):
        with timer(do_print=do_print_timers,text=f"Calc Weights {i}"):

            if ceil(cross_mutation_ratio * generation_limit) <= i:
                population_fitness = np.array([ fitness_func(genome) for genome in population])
            else:
                population_fitness = np.array([ 1 for genome in population])
            # max_fitness = max(population_fitness)
            population_fitness = ((1- population_fitness)+5)**2
            # print(population_fitness)
        if printer is not None:
            printer(population, i, fitness_func)

        next_generation = []

        counter = min(2,len(population))
        # for gen in population:
        #     if fitness_func(gen) > 0.001 and counter and len(population)>2:
        #         next_generation += [population[-counter]]
        #         counter -= 1
        
        next_generation+=list(nsmallest(counter, population, key=lambda x: fitness_func(x)))
        # best_fit = None
        # for gen in population:
        #     print(f"gene fitness ---> {fitness_func(gen)}  Avg: {np.average(gen[:,1])}")
        #     if fitness_func(gen) > 0.001:
        #         if best_fit is None:
        #             best_fit = gen
        #         elif fitness_func(gen) < fitness_func(best_fit):
        #             best_fit = gen
        # next_generation+=[best_fit]
        # print(f"Best fitness ==> {fitness_func(best_fit)} NMI {NMI(best_fit)}")


        with timer(do_print=do_print_timers,text="-----------------"):
            for j in range(len(population)-counter):
                with timer(do_print=do_print_timers,text="\tSelection"):
                    # parents = selection_func(population, fitness_func)
                    parents = selection_pair(population, fitness_func,population_fitness)
                # parent_a = population.pop(randint(0,len(population)-1))
                # parent_b = population.pop(randint(0,len(population)-1)) if len(population) else parent_a[:]
                # print(parents[0])
                # print(parents[1])
                with timer(do_print=do_print_timers,text="\tCrossover"):
                    if crossover_tresholds is None:
                        offspring_a  = crossover_func(parents[0], parents[1])
                    else :
                        
                        thresh = np.average(parents[1][:,1]) + np.average(parents[0][:,1])/2.0
                        # offspring_a  = crossover_func(parents[0], parents[1],threshold=crossover_tresholds[i])
                        offspring_a  = crossover_func(parents[0], parents[1],threshold=thresh)
                # offspring_a  = crossover_func(parents[0], parents[1]) 
                # if ceil(cross_mutation_ratio * generation_limit) <= i:
                #     with timer(do_print=do_print_timers,text="\tMutaion"):
                #         offspring_a = mutation_func(offspring_a)
                next_generation += [offspring_a]
                # next_generation += [b]
        
        population = np.array(next_generation)

        # Filtering
        generation_fitness = np.array([fitness_func(gen) for gen in population])
        average_sim = np.array(list(map(np.average,population[:,:,0])))

        for i in range(len(generation_fitness)):
            if average_sim[i] >= np.average(average_sim):
                population[i]=mutation_func(population[i],probability=np.average(average_sim))

        # if ceil(cross_mutation_ratio * generation_limit) <= i:
        #     posize = len(population)
        #     # print("")
        #     # print(f"before -> {len(population)}")
        #     population = np.array(population)
        #     average_sim = NormalizeData(np.array(list(map(np.var,population[:,:,0]))))
        #     # average_fitness = NormalizeData(np.array([fitness_func(gen) for gen in population]),True)
        #     # population = population[average_sim+average_fitness>=np.average(average_sim+average_fitness)]
        #     population = population[average_sim>=np.average(average_sim)]
        #     # population = population[average_fitness>=np.average(average_fitness)]
        #     # population=list(nsmallest(posize-counter, population, key=lambda x: fitness_func(x)))
        #     after_posize = len(population)

        #     print(f"{posize} -> {after_posize}")
        # population = next_generation
    # best_gene = sorted(population,key=lambda x:fitness_func(x),reverse=True)[0]
    # for _index in range(ceil(generation_limit * (1-cross_mutation_ratio))):
    #     best_gene = mutation_func(best_gene)
    # population = [best_gene]
    
    # print("End of Generation")
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

# ## `Population Initialization`

# ### Random Walk

def create_walks(G:nx.Graph,n_wlaks,walk_len,start_nodes,percentage):
    walks = walker.random_walks(G,n_walks=n_wlaks,walk_len=walk_len,start_nodes=start_nodes)
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

    """
        elif selection_type ==3:
            degs = {}
            for n,d in dico:
                if degs.__contains__(d):
                    degs[d].append(n)
                else:
                    degs[d] = []
            return choices( list(graph.degree((max(list(degs.values()),key=lambda x:len(x))))) ,k=ceil(percentage/100* dico.__len__()))

        elif selection_type ==4:
            degs = {}
            for n,d in dico:
                if degs.__contains__(d):
                    degs[d].append(n)
                else:
                    degs[d] = []
            freq_nodes = choices( list(graph.degree((max(list(degs.values()),key=lambda x:len(x))))) ,k=ceil((percentage/100* dico.__len__())*0.5))
            avg_nodes = sorted (list(graph.degree),key= lambda x:abs(average_degree - x[1]))[:ceil((percentage/100* dico.__len__())*0.25)]
            max_nodes = sorted (list(graph.degree),key= lambda x:abs(max_degree - x[1]))[:ceil((percentage/100* dico.__len__())*0.25)]
            avg_nodes.extend(max_nodes)
            avg_nodes.extend(freq_nodes)
            return avg_nodes


        # return sorted(dico.items(), key=lambda item: item[1],reverse=False)[:ceil(percentage/100* dico.__len__())]
    """

# ## Generate Chromosome

from gen.simmilarity import Simmilarity
# this will give us common neighbours between two nodes

# ### how chromosomes look like:
# karate dataset
# * [(33, 0.083), (33, 0.0), .... ,(33, 0.0), (33, 0.7), `(33, 1)`]
# * We have N communities each starting with one `hub` node assisiated to them, and each indivitual's similiarity values are calculated to the related `hub`
# * in the example above node 33 is `hub` and its similarity to itself is 1

#TODO Add random selection
#TODO Add dbscan & km and ... try to seperate by they degree count
# load(DataSet.football)
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
    
    print(f'lenght of random walks:{len(hub_lines[0])}')
    return genomes

def generate_genome(hub,G:nx.Graph,rw_nodes=None) -> 'Genome':
    return np.array([[hub,Simmilarity.get_simmilarity(G,node,hub,rw_nodes)] for node in G.nodes()])



dataset:Dataset = DatasetLoader.football
_generations  = generate_population(dataset.G, percentage=10,rw_percentage=2,n_wlaks=500,walk_len=4,hub_selection_type=0,remove_repeated_hubs=False)

plt.rcParams['figure.dpi'] = 200

fig, axs = plt.subplots(nrows=3,ncols=4,figsize=(4*4, 4*3))
fig.tight_layout(h_pad=-2,w_pad=-3)
def draw_random_walk_nodes(axs,rw_hubs,labels=False):
    layout = nx.kamada_kawai_layout(dataset.G)

    if labels:
        global colors
    for i in range(len(rw_hubs)):
        node_size =[50]*len(dataset.G.nodes())
        for node,occarance in rw_hubs[i]:
            node_size[node] = 100+occarance/2
        
        node_color =["#666666"]*len(dataset.G.nodes())
        node_color[rw_hubs[i][0][0]] = "#aa6666"
        if labels:
            node_color=[colors[int(_k)] for _k,l in dataset.genome_true_label]
        for j in range(0,5):
            nx.draw_networkx_nodes(dataset.G,ax=axs[i//4][i%4],
                node_color='black',
                node_size=np.array(node_size)+50*(j*0.3),
                            pos=layout,
                            alpha=(5-j)*0.1)
        nx.draw_networkx_nodes(dataset.G,ax=axs[i//4][i%4],
            node_color=node_color,
            node_size=node_size,
                          pos=layout)
        nx.draw_networkx_edges(dataset.G,ax=axs[i//4][i%4], pos=layout, width=.3, alpha=0.5)

        # axs[i,j].set_title(merger.comm_scores.get_score(result_size),color='black')
refresh_colors(dataset.G)
draw_random_walk_nodes(axs,hub_lines,labels=True)

from gen.utils import genome_to_label_converter

def NMI(genome:Genome) -> int:
    return normalized_mutual_info_score (dataset.true_label,genome_to_label_converter(genome))

# ## `Mutation`

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



    # print(redirect_hubs)
    
    # print(get_redirect(0))
    
    # print(get_redirect_similarity(0,hub_similarity))

    offspring= np.array(a,copy=True)

    visite = np.zeros(len(offspring))
    for node_index in range(len(offspring)):
        aa = a[node_index]
        bb = b[node_index]
        hub_in_a = aa[0]
        hub_in_b = bb[0]
        
        # if not redirect_hubs.get(hub_in_a) or not redirect_hubs.get(hub_in_b):
        #     print(a) 
        #     print(b) 

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
            # a_half = hub_similarity[get_redirect(hub_in_a)].get(hub_in_a)
            # b_half = hub_similarity[get_redirect(hub_in_b)].get(hub_in_b)
            # offspring[node_index] = [get_redirect(hub_in_a),np.average([aa[1],a_half])] \
            #                             if np.average([aa[1],a_half]) > np.average([bb[1],b_half]) \
            #                                 else [get_redirect(hub_in_b),np.average([bb[1],b_half])]
        # visite[node_index]= 1
        # if aa[1] >= threshold and bb[1]>= threshold :
        #     if aa[1]==1 or bb[1] == 1:
        #         master = min(aa,bb,key = lambda x:x[1]) 
        #         slave = max(aa,bb,key = lambda x:x[1]) 

    DrawGeneticData.instance.append([a,b,offspring])
    return offspring

# ## `Load Data and Run Evolution`

dataset = DatasetLoader.lfr(100,average_degree=10,max_degree=50,mu=0.1,tau1=2,tau2=1.1,min_community=10,max_community=50,draw=True)

sparsity_score = (1 - nx.density(dataset.G)) * (1 / (len(dataset.G.nodes()) ** 0.5)) * 100
num_nodes = dataset.G.number_of_nodes()
avg_degree = np.average(list(dict(dataset.G.degree()).values()))
num_edges = dataset.G.number_of_edges()

print(f"Sparsity Score     : {sparsity_score:.2f}")
print(f"Number of Nodes    : {num_nodes}")
print(f"Average Node Degree: {avg_degree:.2f}")
print(f"Number of Edges    : {num_edges}")

# $ \frac{1-density}{\sqrt{N}+1} $

from functools import partial

from gen.utils import genome_to_comm_convertor

# dataset = DatasetLoader.football
meu = 1
limit=11
lb   =10
all_nmi = []
# nmi_global = 0
nmi_global = 0
besti = None
best_gene= None
population_percentage = 10#(1-nx.density(G)) * (1/ (len(G.nodes())**0.5 ))*100
g_limit = 10
crossover_thresh = 6
mutation_ratio = 0
DrawGeneticData.instance = []
for g_limit in range(lb,limit):
# for g_limit in [10]*1:

	thresholds = np.linspace(start=0,stop=0,num=g_limit)
	thresholds = thresholds**0.5
# for population_percentage in range(10,20,10):
# for crossover_thresh in range(0,10):
# for mutation_ratio in range(0,10):
	nmi = 0
	for i in range(1):
		with timer(do_print=True):
			population, generations = run_evolution(
				# populate_func=partial(genetic.generate_population, size=10, genome_length=len(things)),
				populate_func=partial(generate_population, G=dataset.G, percentage=population_percentage,rw_percentage=10 ,n_wlaks=500,walk_len=4,hub_selection_type=0,remove_repeated_hubs=False),
				# fitness_func=partial(knapsack.fitness, things=things, weight_limit=weight_limit),
				fitness_func=partial(kkm_rc_fitness,graph = dataset.G),
				# fitness_func=partial(community_fitness,graph = G, weight_limit=weight_limit),
				cross_mutation_ratio=mutation_ratio*0.1,
				fitness_limit=.001,
				generation_limit=g_limit,
				first_crossover_treshold=0.9,
				# crossover_tresholds=[0.8,0.5,0.4,0.2,0.2,0.2],
				# crossover_tresholds=[1,1,1,1,1,1],
				crossover_tresholds=thresholds,
				crossover_func= partial(crossover,graph=dataset.G,lower_bound=-0.01),
				mutation_func = partial(mutation,graph= dataset.G,probability=1.1,),
				do_print_timers=False
			)

		# best_gene = sorted(population,key=lambda x:NMI(x),reverse=True)[0]
		best_gene = sorted(population,key=lambda x:nx.community.modularity(dataset.G,genome_to_comm_convertor(x).values()),reverse=True)[0]
		# best_gene = sorted(population,key=lambda x:kkm_rc_fitness(x,dataset.G),reverse=False)[0]
		if  NMI(best_gene) > nmi :
			nmi = NMI(best_gene)
			if nmi > nmi_global:
				nmi_global = nmi
				besti = best_gene

	all_nmi.append(nmi)


print(f"NMI: {max(all_nmi)}")
# dolp = 0.88
# blogs = .37
# football = .76

# plt.plot(range(1),all_nmi)

plt.plot(range(lb,limit),all_nmi)
# plt.plot(range(10,20,10),all_nmi)
# plt.plot(np.arange(0.0,1.0,0.1),all_nmi)
# plt.plot(np.arange(0.1,1.0,0.1),all_nmi)

# ([kkm_rc_fitness(gen,dataset.G) for gen in population])

mutation_diagram_data = []
besti1 = besti.copy()
print(np.median(besti[:,1]))
for i in np.linspace(0,1.1,20):
    besti1 = mutation(besti1,dataset.G,majaroity_density=0.5,probability=i)
    # print(NMI(besti1,G))
    mutation_diagram_data.append([i,NMI(besti1)])
mutation_diagram_data = np.array(mutation_diagram_data)
plt.plot(mutation_diagram_data[:,0],mutation_diagram_data[:,1])

from gen.utils import draw
# besti_rearanged = besti[:]
# lili = list(zip(TRUE_LABEL,besti))
# # lili
# lili_sorted = sorted(lili,key=lambda x:x[1][1],reverse=True)
# new_labels = {}
# for new,old in lili_sorted:
#     if not new_labels.get(old[0]):
#         if not list(new_labels.values()).__contains__(new):
#             new_labels[old[0]]= new
#         else:
#             new_labels[old[0]] = old[0]

# # new_labels
# # lili_sorted

# for i in range(len(besti_rearanged)):
#     besti_rearanged[i] = (new_labels[ besti_rearanged[i][0] ] , besti_rearanged[i][1])

refresh_colors(dataset.G)
# draw(G,[besti1,GENOME_TRUE_LABEL],n_cols=2,title=[NMI(besti1),NMI(GENOME_TRUE_LABEL)])
draw(dataset.G,[besti,dataset.genome_true_label],n_cols=2,title=[NMI(besti),NMI(dataset.genome_true_label)],colors=colors)

from gen.utils import comm_to_genome_convertor, genome_to_label_dict


coms = {}
for i in range(len(dataset.genome_true_label)):
    
    if not coms.get(dataset.genome_true_label[i][0]):
        coms[dataset.genome_true_label[i][0]]=[i]
    else:
        coms[dataset.genome_true_label[i][0]].append(i)
coms

plt.figure(figsize=(5,5))

# load(DataSet.football)

node_list=list(coms.keys())

label_dict = genome_to_label_dict( comm_to_genome_convertor(coms))
CG = nx.Graph()
genome_community = []
for com in coms.keys():
    # print(com)
    CG.add_node(com)
    genome_community.append((com,1))
# for node,com in label_dict.items():

for u,v in dataset.G.edges():
    if CG.has_edge(label_dict[v],label_dict[u]):
        CG[label_dict[v]][label_dict[u]]['weight']+=1
    else:
        CG.add_edge(label_dict[v],label_dict[u],weight = 1) 
CG_sub = nx.subgraph(CG,node_list)

elarge = [(u, v) for (u, v, d) in CG_sub.edges(data=True) if d["weight"] > 0.5]
edge_labels = nx.get_edge_attributes(CG_sub, "weight")

pos = nx.circular_layout(CG_sub)
nx.draw(CG_sub, pos, node_size=200,with_labels = True)
nx.draw_networkx_edges(CG_sub, pos, width=1)

edge_labels = nx.get_edge_attributes(CG_sub, "weight")
nx.draw_networkx_edge_labels(CG_sub, pos, edge_labels)

from gen.utils import detect_hub_nodes


plt.rcParams["animation.html"] = "jshtml"
# plt.rcParams['figure.dpi'] = 150  
plt.ion()

fig, axs = plt.subplots(figsize=(5, 5))

def animate_hub_creation(t,labels,axs,rw_hubs):
# def draw(G,labels,n_rows=1,n_cols=2,size = 10,title= [""]):
    layout = nx.kamada_kawai_layout(dataset.G)
    dico ={}
    for k,v in detect_hub_nodes(dataset.G):
        dico[k] = k
    k = 0
    # colors = [k for k in range(500)]
    # colors =[k for k,v in mcolors.TABLEAU_COLORS.items()]
    global colors
    # colors =["#ffff00",
    # "#ff00ff",
    # "#00ffff",
    # "#0000ff",
    # "#00ff00",
    # "#ff0000",
    # "#aa44aa",
    # "#44aa44",
    # "#55bbaf"]


    axs.cla()
    node_size =[100]*len(dataset.G.nodes())
    for node,occarance in rw_hubs[t]:
        node_size[node] = 200+occarance

    nx.draw_networkx(dataset.G, ax=axs,
        node_color=[colors[int(_k)] for _k,l in labels],
        node_size=node_size, with_labels=True,
                    font_size=5, font_color='black', pos=layout,width=0.1)
    # axs[i,j].set_title(merger.comm_scores.get_score(result_size),color='black')


# draw(G,[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],n_cols=2)



animation.FuncAnimation(fig, animate_hub_creation, frames=len(hub_lines),fargs=[dataset.genome_true_label,axs,hub_lines])

# ## `Draw Genetic Results`

# plt.rcParams["animation.html"] = "jshtml"
# plt.rcParams['figure.dpi'] = 100  
# plt.ion()


# fig, axs = plt.subplots(nrows=1,ncols=3, figsize=(5*3, 5))
# def animate(t,labels,axs,title):



#     n_rows=1
#     n_cols =3
# # def draw(G,labels,n_rows=1,n_cols=2,size = 10,title= [""]):
#     layout = nx.kamada_kawai_layout(dataset.G)
#     dico ={}
#     for k,v in detect_hub_nodes(dataset.G):
#         dico[k] = k
#     k = 0
#     # colors = [k for k in range(500)]
#     # colors =[k for k,v in mcolors.TABLEAU_COLORS.items()]
#     global colors
#     # colors =["#ffff00",
#     # "#ff00ff",
#     # "#00ffff",
#     # "#0000ff",
#     # "#00ff00",
#     # "#ff0000",
#     # "#aa44aa",
#     # "#44aa44",
#     # "#55bbaf"]

#     for i in range(n_rows):
#         for j in range(n_cols):

#             axs[j].cla()
#             # hub = labels[t][k].index(max(labels[t][k],key=lambda x:x[1]))
#             # node_size = [150] * len(G.nodes())
#             # node_size[hub] = 450
#             # print()
#             node_size = 100+(20**(1+np.array([b for a,b in labels[t][k]])))

#             nx.draw_networkx(dataset.G, ax=axs[j],
#              node_color=[colors[int(_k)] for _k,l in labels[t][k]],
#              node_size=node_size, with_labels=True,
#                             font_size=13, font_color='black', pos=layout)
#             axs[j].set_title(kkm_rc_fitness(labels[t][k],dataset.G) ,color = 'black')
#             k+=1
#             # axs[i,j].set_title(merger.comm_scores.get_score(result_size),color='black')


# # draw(G,[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# # [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],n_cols=2)

# colors = []
# n = len(dataset.G.nodes())
# for i in range(n):
#     colors.append('#%06X' % randint(0, 0xFFFFFF))

# animation.FuncAnimation(fig, animate, frames=len(DrawGeneticData.instance),fargs=[DrawGeneticData.instance,axs,['parent_a','parent_b','ofspring']])
