from matplotlib import pyplot as plt
import scipy.io as io
import networkx as nx
from gen.utils import label_to_genome_converter

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
                # print(node)
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
        # comms = dict(zip(list(set([c for c,p in labels[k]])),colors))

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
    karate = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/karate/dataset-karate.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/karate/comm-karate.mat')['factCommunities']))
    football =_load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/football/1st/dataset-football.mat')['N'],
                         io.loadmat('../Datasets/Non-Overlap Unweighted/football/1st/comm-total-football.mat')['commDD']))
    books1 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/book/1st/dataset-book')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/book/1st/comm-total-book')['commDD']))
    # books2 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/book/2nd/dataset-book')['N'],
                        # io.loadmat('../Datasets/Non-Overlap Unweighted/book/2nd/comm-total-book')['commDD')])
    dolphins1 = _load(_calc_comm(io.loadmat('../Datasets/Non-Overlap Unweighted/dolphin/1st/dataset-dolphin')['N'],
                          io.loadmat('../Datasets/Non-Overlap Unweighted/dolphin/1st/comm-total-dolphin')['commDD']))                        
    dolphins2 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/dolphin/2nd/dataset-dolphin')['N'],
                           io.loadmat('../Datasets/Non-Overlap Unweighted/dolphin/2nd/comm-dolphin')['V']))
    blogs = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/blogs/dataset-blogs')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/blogs/comm-blogs')['V']))
    blogs = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/blogs/dataset-blogs')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/blogs/comm-blogs')['V']))
    lfr1 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f1.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f1.mat')['V']))
    lfr2 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f2.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f2.mat')['V']))
    lfr3 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f3.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f3.mat')['V']))
    lfr4 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f4.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f4.mat')['V']))
    lfr5 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f5.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f5.mat')['V']))
    lfr6 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f6.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f6.mat')['V']))
    lfr7 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f7.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f7.mat')['V']))
    lfr8 = _load(_calc_comm( io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/dataset-f8.mat')['N'],
                        io.loadmat('../Datasets/Non-Overlap Unweighted/LFR/comm-f8.mat')['V']))
    

class DatasetOptions():
    Amazon=0
    DBLP=1
    # Youtube=2


class LargeDataset():
    def load_data(self,choise:DatasetOptions, want_best = False):
        datasets = [
            {
                # Amazon
                'adj':"../Datasets/Overlap-unweighted/com-Amazon/com-Amazon.mtx",
                'community':"../Datasets/Overlap-unweighted/com-Amazon/com-Amazon_Communities_all.mtx",
                'best_community':"../Datasets/Overlap-unweighted/com-Amazon/com-Amazon_Communities_top5000.mtx"    
            },
            {
                'adj':'../Datasets/com-DBLP/com-DBLP/com-DBLP.mtx',
                'community':'../Datasets/com-DBLP/com-DBLP/com-DBLP_Communities_all.mtx',
                'best_community':'../Datasets/com-DBLP/com-DBLP/com-DBLP_Communities_top5000.mtx',
            }
            ]
        
        data_adj = io.mmread(datasets[choise]['adj'])
        data_communities = io.mmread(datasets[choise]['best_community']) if want_best else  io.mmread(datasets[choise]['community'])
        
        return data_adj,data_communities

    def convert(self,data_adj,data_communities):
        G =nx.from_scipy_sparse_array(data_adj)
        self.G = G
        self.edges = {}
        for v,u_s in G.adj.items():
            self.edges[v]= list(u_s)
                 

        nodes,communities =  data_communities.nonzero()
        self.communities = {}
        self.nodes = {}
        for i in range(len(communities)):
            if self.communities.get(communities[i])==None:
                self.communities[communities[i]] = [nodes[i]]
            else:
                self.communities[communities[i]].append( nodes[i] )
            
            if self.nodes.get(nodes[i])==None:
                self.nodes[nodes[i]] = [communities[i]]
            else:
                self.nodes[nodes[i]].append(communities[i])




    def __init__(self,choise:DatasetOptions,want_best=False):
        data_adj,data_communities = self.load_data(choise,want_best)

        self.communities = {}
        self.G = None       
        self.edges = {}
        self.nodes = {}
        self.communities = {}
        self.convert(data_adj,data_communities)