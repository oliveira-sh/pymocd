import networkx as nx
import numpy as np
class Simmilarity:
    def __init__(self, G: nx.Graph):
        self.v_u_similarity = {}
        self.v_u_common = {}
        self.calculate_common_neighbours(G)

    @staticmethod
    def get_sim_value(G,v,u,alpha=0.5):
        sim = list(nx.common_neighbors(G, v, u))
        neighbors_value = len(sim)
        return alpha * neighbors_value/len(list(G.neighbors(v))) + (1-alpha) * G.has_edge(v, u)

    @staticmethod
    # TODO : hub should ne a list for merging section
    def get_simmilarity(G: nx.Graph, v, hub,rw_nodes=None, alpha=0.5):

        # print(f"aaaaaaaa ------> {v}")
        if v == hub:
            return 1
        if len(list(G.neighbors(v))) == 0:
            return 0
        avg = np.average([Simmilarity.get_sim_value(G,v,node) for node,occurance in rw_nodes],weights=rw_nodes[:,1])
        # print(f"->{avg}")
        return avg# sum(sim_i * (occurance_i/ max(occurance)) / len(rw_walks) 
        # (2:30)   -> 0.9
        # (1: 15)  -> 0.8
        # (3: 10)  -> 0.3

    def calculate_common_neighbours(self, G: nx.Graph):
        for v in G.nodes():
            if self.v_u_similarity.get(v) is None:
                self.v_u_similarity[v] = []
                self.v_u_common[v] = {}
            for u in G.neighbors(v):
                sim = list(nx.common_neighbors(G, v, u))
                # we can use heap to get top similar nodes with sorting
                self.v_u_common[v][u] = sim
                if self.v_u_common.get(u) is None:
                    self.v_u_common[u] = {v: sim}
                self.v_u_similarity[v].append((u, len(sim)))

                if u in self.v_u_similarity:
                    self.v_u_similarity[u].append((v, len(sim)))
                else:
                    self.v_u_similarity[u] = [(v, len(sim))]

        for v in self.v_u_similarity:
            self.v_u_similarity[v].sort(key=lambda x: x[1], reverse=True)

    def choose_best(self, v, k=0):
        """
        k: k'th most similar node  

        >>> {
        0:
            [(1, 7),
            (2, 5),
            (3, 5)],
        1:
            [(2, 4),
            (3, 4)] 
        }

        >>> choose(0) -> (1,7) 
        """
        if self.v_u_similarity.get(v) is not None:
            if len(self.v_u_similarity[v]) > k:
                return self.v_u_similarity[v][k]
            return None

    def choose_all(self, v):
        """
        v: return all related to v
        """
        if self.v_u_similarity.get(v) is not None:
            return self.v_u_similarity[v]
        return None
