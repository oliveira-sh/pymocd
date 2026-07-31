# Benchmarks

Every documentation deployment re-runs a bounded benchmark suite in CI and
regenerates the figures on this page. Three experiment types are included:
a **synthetic LFR sweep** over the mixing parameter *μ*, a comparison on
**real networks** with well-known community structure, and two
**algorithm parameter** studies (number of generations and Pareto fronts).
The CI workload is deliberately reduced — fewer runs and smaller graphs than
the full local suite (`make benchmark`) — so read these plots as indicative
trends rather than publication-grade numbers. Each figure is also exported as
PNG and PDF next to the embedded SVG under `assets/benchmarks/`.

## Real networks

All detectors run on Zachary's Karate Club, Les Misérables, and the
Florentine families graphs, averaged over repeated seeds. Raw data:
[real_nets.csv](assets/benchmarks/real_nets/real_nets.csv).

### Modularity

![Modularity on real networks](assets/benchmarks/real_nets/real_nets_modularity.svg)

Higher bars mean a stronger community structure found on that network; error bars show the spread across seeds.

### NMI

![NMI on real networks](assets/benchmarks/real_nets/real_nets_nmi.svg)

Normalized mutual information against the known ground truth (only networks with labels appear); closer to 1 is a closer match.

### AMI

![AMI on real networks](assets/benchmarks/real_nets/real_nets_ami.svg)

Adjusted mutual information corrects NMI for chance agreement, so it is the stricter of the two label-recovery scores.

### Runtime

![Runtime on real networks](assets/benchmarks/real_nets/real_nets_time.svg)

Wall-clock seconds per detection on a log scale; lower bars are faster.

## Synthetic LFR sweep

Each detector runs across LFR benchmark graphs with increasing mixing
parameter *μ* (higher *μ* means blurrier, harder-to-detect communities).
Raw data: [lfr_mu.csv](assets/benchmarks/lfr_mu.csv), with an incremental
backup in [lfr_mu_bk.csv](assets/benchmarks/lfr_mu_bk.csv).

### Comparison matrix

![Comparison matrix](assets/benchmarks/mu/comparison_matrix.svg)

All four metrics side by side per algorithm and *μ*; scan a row to see how gracefully each detector degrades as mixing increases.

### Performance scorecard

![Performance scorecard](assets/benchmarks/mu/performance_scorecard.svg)

An aggregate ranking across the sweep; higher composite scores mean better overall quality/runtime trade-offs.

### NMI vs μ

![NMI vs mu](assets/benchmarks/mu/nmi_plot.svg)

Ground-truth recovery as communities blur; flatter curves that stay high are more robust detectors.

### AMI vs μ

![AMI vs mu](assets/benchmarks/mu/ami_plot.svg)

The chance-adjusted counterpart of the NMI curve; it penalizes trivially fragmented or merged partitions.

### Modularity vs μ

![Modularity vs mu](assets/benchmarks/mu/modularity_plot.svg)

Modularity of the returned partition; it naturally decreases with *μ* since the planted structure itself weakens.

### Runtime vs μ

![Runtime vs mu](assets/benchmarks/mu/time_plot.svg)

Detection time across the sweep; watch for algorithms whose cost grows as communities get harder to separate.

## Algorithm parameters

### Community evolution across generations

![Community evolution overview](assets/benchmarks/community_evolution_overview.svg)

Six snapshots of the same LFR graph as `pymocd.scale` runs for more generations; colors are communities, so watch fragmented labels consolidate into the planted structure.

### Generation 10

![Communities at generation 10](assets/benchmarks/communities_gen_10.svg)

The earliest snapshot: the population has barely evolved, so partitions are still noisy and over-fragmented.

### Generation 30

![Communities at generation 30](assets/benchmarks/communities_gen_30.svg)

Small fragments begin merging as selection pressure favors coherent groups.

### Generation 50

![Communities at generation 50](assets/benchmarks/communities_gen_50.svg)

The dominant communities are recognizable, with disagreement left mostly at the boundaries.

### Generation 80

![Communities at generation 80](assets/benchmarks/communities_gen_80.svg)

Refinement stage: boundary nodes settle and community counts stabilize.

### Generation 100

![Communities at generation 100](assets/benchmarks/communities_gen_100.svg)

Close to convergence; compare with generation 110 to judge whether extra generations still pay off.

### Generation 110

![Communities at generation 110](assets/benchmarks/communities_gen_110.svg)

The final snapshot; if it is indistinguishable from generation 100, the run has converged.

### Pareto front

![Pareto front](assets/benchmarks/pareto_front_plot.svg)

Each point is one non-dominated partition in intra/inter objective space, colored by its Q score; the star marks the max-Q solution the API returns.

### Pareto frontier analysis

![Pareto frontier analysis](assets/benchmarks/pareto_frontier_analysis.svg)

The same front alongside how Q relates to community count, modularity, NMI, and AMI; the dashed line marks the selected solution's Q in every panel.
