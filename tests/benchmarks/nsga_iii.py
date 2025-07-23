import numpy as np
import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from typing import List, Dict, Any


LARGE = 1e6  # penalty value for trivial partitions


class CommunityDetectionProblem(Problem):
    """
    Multi-objective community detection problem using NSGA-III
    """

    def __init__(self, graph: nx.Graph, variant: str = "KRM") -> None:
        if variant not in {"KRM", "CCM"}:
            raise ValueError(f"Unknown variant '{variant}'. Choose 'KRM' or 'CCM'.")
        self.graph = graph
        self.variant = variant
        self.n_nodes = graph.number_of_nodes()
        self.node_list = list(graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        # Precompute adjacency & degrees
        self.neighbors: Dict[int, List[int]] = {}
        for i, node in enumerate(self.node_list):
            nbrs = [self.node_to_idx[nbr] for nbr in graph.neighbors(node)]
            self.neighbors[i] = nbrs + [i]  # allow self-loop to isolate

        self.degrees = {i: graph.degree(self.idx_to_node[i]) for i in range(self.n_nodes)}

        n_obj = 3
        xl = np.zeros(self.n_nodes, int)
        xu = np.full(self.n_nodes, self.n_nodes - 1, int)
        super().__init__(n_var=self.n_nodes, n_obj=n_obj, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, X: np.ndarray, out: Dict[str, Any], *args, **kwargs) -> None:
        pop_size = X.shape[0]
        F = np.full((pop_size, self.n_obj), LARGE, dtype=float)

        for i, x in enumerate(X):
            x = x.astype(int)
            comms = self._decode(x)
            if len(comms) <= 1:
                # leave F[i] = [LARGE, LARGE, LARGE]
                continue

            if self.variant == "KRM":
                F[i, 0] = self._kernel_k_means(comms)
                F[i, 1] = self._ratio_cut(comms)
                F[i, 2] = -self._modularity(comms)
            else:  # CCM
                F[i, 0] = -self._community_fitness(comms)
                F[i, 1] = -self._community_score(comms)
                F[i, 2] = -self._modularity(comms)

        out["F"] = F

    def _decode(self, sol: np.ndarray) -> List[List[int]]:
        parent = list(range(self.n_nodes))

        def find(u: int) -> int:
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u: int, v: int) -> None:
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv

        for i, gene in enumerate(sol):
            if gene in self.neighbors[i]:
                union(i, gene)

        comms: Dict[int, List[int]] = {}
        for i in range(self.n_nodes):
            r = find(i)
            comms.setdefault(r, []).append(i)
        return list(comms.values())

    def _modularity(self, comms: List[List[int]]) -> float:
        # convert to node-label lists
        node_groups = [[self.idx_to_node[i] for i in c] for c in comms]
        return nx_modularity(self.graph, node_groups)

    def _kernel_k_means(self, comms: List[List[int]]) -> float:
        n, m = self.n_nodes, len(comms)
        total = 0.0
        for c in comms:
            if not c: continue
            internal = sum(self._internal_deg(i, c) for i in c)
            total += internal / len(c)
        return 2 * (n - m) - total

    def _ratio_cut(self, comms: List[List[int]]) -> float:
        total = 0.0
        for c in comms:
            if not c: continue
            external = sum(self._external_deg(i, c) for i in c)
            total += external / len(c)
        return total

    def _community_fitness(self, comms: List[List[int]]) -> float:
        α = 1.0
        total = 0.0
        for c in comms:
            for i in c:
                k_in = self._internal_deg(i, c)
                k_out = self._external_deg(i, c)
                k_tot = k_in + k_out
                if k_tot > 0:
                    total += k_in / (k_tot ** α)
        return total

    def _community_score(self, comms: List[List[int]]) -> float:
        r = 1.0
        total = 0.0
        for c in comms:
            if len(c) < 2:
                continue
            mu = [self._internal_deg(i, c) / (len(c) - 1) for i in c]
            M = float(np.mean(mu)) if r == 1 else (np.mean([m**r for m in mu])) ** (1/r)
            V = self._internal_edges(c)
            total += M * V
        return total

    def _internal_deg(self, node: int, comm: List[int]) -> int:
        cnt = 0
        s = set(comm)
        for nbr in self.graph.neighbors(self.idx_to_node[node]):
            j = self.node_to_idx[nbr]
            if j in s and j != node:
                cnt += 1
        return cnt

    def _external_deg(self, node: int, comm: List[int]) -> int:
        cnt = 0
        s = set(comm)
        for nbr in self.graph.neighbors(self.idx_to_node[node]):
            j = self.node_to_idx[nbr]
            if j not in s:
                cnt += 1
        return cnt

    def _internal_edges(self, comm: List[int]) -> int:
        cnt = 0
        s = set(comm)
        for u in comm:
            for nbr in self.graph.neighbors(self.idx_to_node[u]):
                v = self.node_to_idx[nbr]
                if v in s and u < v:
                    cnt += 1
        return cnt

    def communities_to_labels(self, comms: List[List[int]]) -> Dict[Any, int]:
        labels = {}
        for cid, c in enumerate(comms):
            for i in c:
                labels[self.idx_to_node[i]] = cid
        return labels


class DuplicateFilter(Callback):
    """
    Replace trivial all-in-one solutions each generation
    """
    def __init__(self, problem: CommunityDetectionProblem) -> None:
        super().__init__()
        self.problem = problem

    def notify(self, algorithm) -> None:
        if not hasattr(algorithm, "pop") or algorithm.pop is None:
            return
        for ind in algorithm.pop:
            comms = self.problem._decode(ind.X.astype(int))
            if len(comms) <= 1:
                # Resample
                new = np.array([np.random.choice(self.problem.neighbors[j])
                                for j in range(self.problem.n_nodes)], dtype=int)
                ind.X = new
                ind.F = None
                ind.evaluated = False


class NeighborSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), int)
        for i in range(n_samples):
            for j in range(problem.n_var):
                X[i, j] = np.random.choice(problem.neighbors[j])
        return X


def run_nsga3(graph: nx.Graph, variant: str = "KRM",
              pop_size: int = 100, n_gen: int = 100):
    problem = CommunityDetectionProblem(graph, variant)
    ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
    algo = NSGA3(
        pop_size=pop_size,
        ref_dirs=ref_dirs,
        sampling=NeighborSampling(),
        crossover=TwoPointCrossover(),
        mutation=PolynomialMutation(eta=20),
        eliminate_duplicates=True,
        callback=DuplicateFilter(problem)
    )
    res = minimize(problem, algo, ("n_gen", n_gen), verbose=False)
    return res, problem


def select_best(res, problem: CommunityDetectionProblem) -> Dict[Any, int]:
    best_mod, best_comms = -np.inf, None
    for sol in getattr(res, "X", []):
        comms = problem._decode(sol.astype(int))
        if len(comms) <= 1:
            continue
        m = problem._modularity(comms)
        if m > best_mod:
            best_mod, best_comms = m, comms
    if best_comms is None:
        # fallback: each node its own community
        return {n: i for i, n in enumerate(problem.node_list)}
    return problem.communities_to_labels(best_comms)


def run_krm(G: nx.Graph, pop_size=100, n_gen=200) -> Dict[Any, int]:
    res, prob = run_nsga3(G, "KRM", pop_size, n_gen)
    return select_best(res, prob)


def run_ccm(G: nx.Graph, pop_size=100, n_gen=200) -> Dict[Any, int]:
    res, prob = run_nsga3(G, "CCM", pop_size, n_gen)
    return select_best(res, prob)


if __name__ == "__main__":
    G = nx.karate_club_graph()
    print("KRM →", len(set(run_krm(G).values())), "communities")
    print("CCM →", len(set(run_ccm(G).values())), "communities")
    try:
        # Compare with Louvain if available
        from networkx.algorithms.community import louvain_communities
        l = louvain_communities(G)
        print("Louvain →", len(l), "communities")
    except ImportError:
        pass
