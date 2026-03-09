from tqdm import tqdm
import pymocd
import community as community_louvain
import igraph as ig

from .registry import algorithm, _safe, _with_seed
from .objectives import make_motif_average_degree, make_motif_conductance

MIN_MU = 0.1
MAX_MU = 0.7
STP_MU = 0.1
NUM_ND = 10_000

NUM_RUNS = 1
DEBUG = True


def _make_hpmocd(G, name, objective_factories=None, pop_size=100, num_gens=100):
    objectives = [f(G) for f in objective_factories] if objective_factories else []
    model = pymocd.HpMocd(
        G,
        debug_level=0,
        pop_size=pop_size,
        num_gens=num_gens,
        objectives=objectives,
    )
    if DEBUG:
        bar = tqdm(total=model.num_gens, desc=name, leave=False, unit="gen")

        def _on_gen(gen, num_gens, front_size):
            bar.set_postfix(front=front_size)
            bar.update(1)
            if gen == num_gens - 1:
                bar.close()

        model.set_on_generation(_on_gen)
    return model


PY_OBJ_POP_SIZE = 100
PY_OBJ_NUM_GENS = 200


@algorithm("HPMOCD", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def hpmocd_algorithm(G):
    return _make_hpmocd(G, "HPMOCD").run()


@algorithm("HPMOCD-II", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def hpmocd_motif_algorithm(G):
    return _make_hpmocd(
        G,
        "HPMOCD-II",
        [make_motif_average_degree, make_motif_conductance],
        pop_size=PY_OBJ_POP_SIZE,
        num_gens=PY_OBJ_NUM_GENS,
    ).run()


@algorithm("Louvain", needs_conversion=False, parallel=True)
@_safe
@_with_seed
def louvain_algorithm(G):
    return community_louvain.best_partition(G)


@algorithm("Leiden", needs_conversion=False, parallel=True)
@_safe
@_with_seed
def leiden_algorithm(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    ig_graph = ig.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    partition = ig_graph.community_leiden(objective_function="modularity")
    return {nodes[i]: partition.membership[i] for i in range(ig_graph.vcount())}
