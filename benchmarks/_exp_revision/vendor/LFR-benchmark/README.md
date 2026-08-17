# LFR-benchmark

The LFR benchmark model for networks. This repository contains a Python wrapper for the original code in C ([original code](https://sites.google.com/site/andrealancichinetti/)). I'd like to thank the authors for generously allowing me to create this repository. If you use this code, please cite the following paper:

```
@article{PhysRevE.78.046110,
  title = {Benchmark graphs for testing community detection algorithms},
  author = {Lancichinetti, Andrea and Fortunato, Santo and Radicchi, Filippo},
  journal = {Phys. Rev. E},
  volume = {78},
  issue = {4},
  pages = {046110},
  numpages = {5},
  year = {2008},
  month = {Oct},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevE.78.046110},
  url = {https://link.aps.org/doi/10.1103/PhysRevE.78.046110}
}
```

## How to install

To install the LFR-benchmark, follow these steps:

```bash
git clone https://github.com/skojaku/LFR-benchmark
cd LFR-benchmark
python setup.py build
pip install -e .
```

## Usage

Once you have installed the LFR-benchmark, you can use it in your Python code as follows:

```python
import lfr

params = {
    "N": 1000,     # number of nodes
    "k": 50,       # average degree
    "maxk": 100,   # maximum degree
    "minc": 20,    # minimum community size
    "maxc": 100,   # maximum community size
    "tau": 3.0,    # degree exponent
    "tau2": 2.0,   # community size exponent
    "mu": 0.5,     # mixing rate
}

ng = lfr.NetworkGenerator()
data = ng.generate(**params)

net = data["net"]                  # scipy.csr_sparse matrix
community_table = data["community_table"]  # pandas DataFrame
seed = data["seed"]                # Seed value
```

In the code snippet above, you can modify the `params` dictionary to adjust the parameters for network generation according to your needs. After generating the benchmark network using the `ng.generate()` method, you can access the generated network (`net`) as a scipy.csr_sparse matrix, the community information (`community_table`) as a pandas DataFrame, and the seed value (`seed`) used for network generation.
