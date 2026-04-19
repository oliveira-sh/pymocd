from tqdm import tqdm
import pymocd
import community as community_louvain
import igraph as ig

from .registry import algorithm, _safe, _with_seed
from .constants import MIN_MU, MAX_MU, STP_MU, NUM_ND, NUM_RUNS, DEBUG


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


@algorithm("HPMOCD", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def hpmocd_algorithm(G):
    return _make_hpmocd(G, "HPMOCD").run()


def _make_prism(
    G,
    name,
    swarm_size=100,
    num_gens=100,
    archive_cap=100,
    mut_rate=0.1,
    turbulence_frac=0.1,
):
    model = pymocd.Prism(
        G,
        debug_level=0,
        swarm_size=swarm_size,
        num_gens=num_gens,
        archive_cap=archive_cap,
        mut_rate=mut_rate,
        turbulence_frac=turbulence_frac,
    )
    if DEBUG:
        bar = tqdm(total=model.num_gens, desc=name, leave=False, unit="gen")

        def _on_gen(gen, num_gens, archive_size):
            bar.set_postfix(archive=archive_size)
            bar.update(1)
            if gen == num_gens - 1:
                bar.close()

        model.set_on_generation(_on_gen)
    return model


@algorithm("PRISM", needs_conversion=False, parallel=False)
@_safe
@_with_seed
def prism_algorithm(G):
    return _make_prism(G, "PRISM").run()


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
