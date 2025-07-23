from contextlib import contextmanager
import time
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import numpy as np
from random import choices
import numpy as np
import matplotlib.pyplot as plt

@contextmanager
def timer(do_print=True,text=""):
    start = time.time()
    yield
    end = time.time()
    if do_print :
        print(f"{end-start}  {text}")


def genome_to_comm_convertor(genome):
    comms = {}
    i = 0
    for c,s in genome:
        if comms.__contains__(c):
            comms[c].append(i)
        else:
            comms[c] = [i]
        i+=1
    return comms

def label_to_genome_converter(labels):
    return [(item,1) for item in labels]
def comm_to_genome_convertor(dico_of_comms):
    genome = {}
    for c,list_of_nodes in dico_of_comms.items():
        for node in list_of_nodes:
            genome[node]=(c,1)
    return [c for n,c in sorted(list((genome.items())),key=lambda x:x[0])]
def genome_to_label_dict(genome):
    return {index:genome[index][0] for index in range(len(genome))}

def genome_to_label_converter(genome):
    return [genome[index][0] for index in range(len(genome))]


def detect_hub_nodes(graph: nx.Graph, percentage=1,selection_type = 0):
    dico =graph.degree()
    from math import ceil
    average_degree = np.average(np.array(list(graph.degree))[:,1])
    max_degree = np.max(np.array(list(graph.degree))[:,1])
    if selection_type == 0:
        return sorted (list(graph.degree),key= lambda x:abs(average_degree - x[1]))[:ceil(percentage/100* dico.__len__())]

    elif selection_type ==1:
        return sorted (list(graph.degree),key= lambda x:abs(max_degree - x[1]))[:ceil(percentage/100* dico.__len__())]

    elif selection_type ==2:
        avg_nodes = sorted (list(graph.degree),key= lambda x:abs(average_degree - x[1]))[:ceil((percentage/100* dico.__len__())/2)]
        max_nodes = sorted (list(graph.degree),key= lambda x:abs(max_degree - x[1]))[:ceil((percentage/100* dico.__len__())/2)]
        avg_nodes.extend(max_nodes)
        return avg_nodes

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

def draw(G,labels,n_rows=1,n_cols=1,size = 10,title= [""],rw_hubs=None,colors=None):
    fig, axs = plt.subplots(nrows=n_rows,ncols=n_cols, figsize=(size*n_cols, size*n_rows))
    fig.tight_layout(h_pad=-2,w_pad=-3)
    layout = nx.kamada_kawai_layout(G)
    # layout = nx.spring_layout(G)
    dico ={}
    for k,v in detect_hub_nodes(G):
        dico[k] = k
    k = 0

    SIZE = 580
    node_size = SIZE

    for i in range(n_rows):
        for j in range(n_cols):
            
            if rw_hubs :
                node_size =[SIZE]*len(G.nodes())
                for node,occarance in rw_hubs[k]:
                    node_size[node] = SIZE*1.5+occarance

            ax = axs[j] if n_cols + n_rows > 2 else axs
            
            nx.draw_networkx(G, ax=ax,
             node_color=[colors[int(_k)] for _k,l in labels[k]],
             node_size=node_size, with_labels=True,
                            font_size=13, font_color='black', pos=layout,width=0.4)
            ax.set_title(title[k] ,color = 'black')
            k+=1
