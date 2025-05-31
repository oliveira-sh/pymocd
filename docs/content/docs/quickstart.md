---
weight: 100
date: "2023-05-03T22:37:22+01:00"
draft: false
author: "Guilherme Oliveira"
title: "Quickstart"
icon: "rocket_launch"
toc: true
description: "A quickstart guide to downloading and installing pymocd library"
publishdate: "2025-02-01T22:37:22+01:00"
tags: ["Beginners"]
---

## Requirements

- **Python ≥ v3.9**
- **pip**
- **Python venv** (Recommended but optional)

## Install pymocd

{{< tabs tabTotal="2">}}
{{% tab tabName="Linux" %}}

Most Linux distributions come with **python** and **pip** pre-installed. Create a virtual environment and install the library:
```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install pymocd
```

{{% /tab %}}
{{% tab tabName="Homebrew (macOS)" %}}

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install pymocd
```

{{% /tab %}}
{{< /tabs >}}

{{< alert text="We recommend installing `networkx` and `matplotlib` alongside pymocd for full functionality!" />}}

```bash
$ pip install networkx matplotlib
```

## Compiling from Source

For the latest version, you can compile pymocd from the GitHub repository's main branch using **maturin**.

First, clone the repository:

```bash
$ git clone https://github.com/oliveira-sh/pymocd/ && cd pymocd
```

Create a virtual environment for compilation:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
```

Then compile the project:

```bash
$ maturin develop --release
```

## Citation

{{% alert context="info" text="**Note**: Has our library helped your research? Please cite us!" /%}}

```bib
@article{santos2025hpmocd,
  author = {Guilherme Oliveira Santos, Carlos H. G. Ferreira and Gladston J. P. Moreira},
  title = {HP-MOCD: A High-Performance Multi-Objective Community Detection Algorithm for Large-Scale Networks},
  journal = {XXXX},
  year = {2025},
  volume = {X},
  number = {X},
  pages = {XX-XX},
  doi = {XX.XXXX/XXXXXX},
  abstract = {Community detection in social networks has traditionally been approached as a single-objective optimization problem, with various heuristics targeting specific community-defining metrics. However, this approach often proves inadequate for capturing the multifaceted nature of communities. We introduce HP-MOCD, a fully parallelized, evolutionary high-performance multi-objective community detection algorithm designed specifically for large-scale networks. Our implementation overcomes the computational challenges that typically limit multi-objective approaches in this domain. While performance may decrease with networks containing high proportions of inter-community edges, extensive evaluations on synthetic datasets demonstrate that HP-MOCD achieves an exceptional balance between scalability and detection accuracy. Available as open-source software, HP-MOCD offers researchers and practitioners a practical, powerful solution for complex network analysis, particularly for applications requiring both efficiency and detection quality.},
  keywords = {community detection, complex networks, evolutionary algorithms, genetic algorithms, multi-objective}
}
```