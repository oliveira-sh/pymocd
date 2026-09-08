# Algorithms

One directory per community detector. Each is self-contained: it owns its
representation, objectives, operators, search engine and entry point, and is
documented by its own `README.md` alongside the code.

| Module | Method | Paper | Year |
|---|---|---|---|
| [`ccm/`](ccm/README.md) | NSGA-III-CCM | Shaik, Ravi & Deb, *SN Computer Science* 2:13 | 2021 |
| [`cdrme/`](cdrme/README.md) | CDRME | Dabaghi-Zarandi, Afkhami & Ashoori, *Journal of Network and Computer Applications* 234:104070 | 2025 |
| [`gdpso/`](gdpso/README.md) | Greedy Discrete PSO | Cai, Gong, Ma, Ruan, Yuan & Jiao, *Information Sciences* 316:503–516 | 2015 |
| [`hpmocd/`](hpmocd/README.md) | HP-MOCD | Santos et al., *Social Network Analysis and Mining* 15 | 2025 |
| [`krm/`](krm/README.md) | NSGA-III-KRM | Shaik, Ravi & Deb, *SN Computer Science* 2:13 | 2021 |
| [`mmcomo/`](mmcomo/README.md) | MMCoMO | Zhang, Yang, Yang & Zhang, *IEEE Computational Intelligence Magazine* | 2023 |
| [`mocd/`](mocd/README.md) | Shi-MOCD (MOCD-Q, MOCD-D) | Shi, Yan, Cai & Wu, *Applied Soft Computing* 12(2):850–859 | 2012 |
| [`moganet/`](moganet/README.md) | MOGA-Net | Pizzuti, *IEEE ICTAI* / *IEEE TEC* 16(3):418–430 | 2009 |
| [`rimpso/`](rimpso/README.md) | RIMPSO | Santos, in preparation | 2026 |

Each module README carries the full citation, the objective equations, the
representation, the search loop, the parameter table, whether the detector is
bit-deterministic, and every known divergence from its paper.
