# Detectors

Every detector takes a graph and returns a partition as `dict[node, community]`. Isolated nodes are assigned community `-1`.

Two of these are this library's own algorithms — `mr_mocd` and `hpmocd`. The other eight entry points re-implement published methods by other authors; see [Algorithms](../algorithms.md) for the paper, the selection rule and the original implementation (where the authors released one) behind each.

## This library's algorithms

::: pymocd.mr_mocd

::: pymocd.hpmocd

!!! note "Tunable HP-MOCD"
    `hpmocd` takes the graph and nothing else: it runs at the published configuration (`pop_size=100`, `num_gens=100`, `cross_rate=0.7`, `mut_rate=0.5`). The `pymocd.HpMocd` class exposes the same search with those four as constructor arguments, plus `set_objectives` for plugging in your own Python objective functions and `set_on_generation` for a per-generation callback.

## Re-implemented baselines

::: pymocd.cdrme

::: pymocd.mmcomo

::: pymocd.ccm

::: pymocd.krm

::: pymocd.gdpso

::: pymocd.mocd_q

::: pymocd.mocd_d

::: pymocd.moga_net

## Deprecated aliases

`pymocd.scale` is `pymocd.mr_mocd` and `pymocd.scale_fronts` is [`pymocd.mr_mocd_fronts`](fronts.md#pymocd.mr_mocd_fronts), kept from earlier names of this detector. Use the new names.
