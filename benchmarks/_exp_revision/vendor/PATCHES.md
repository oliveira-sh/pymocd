# Vendored LFR reference generator

`LFR-benchmark/` is the original Lancichinetti-Fortunato-Radicchi
`binary_networks` C++ program, taken from
<https://github.com/skojaku/LFR-benchmark> (a mirror of the authors' code at
<https://sites.google.com/site/andrealancichinetti/software>), GPL-2.
Reference: Lancichinetti, Fortunato, Radicchi, *Benchmark graphs for testing
community detection algorithms*, Phys. Rev. E 78, 046110 (2008).

Removed from the checkout: `.git/`, `test/`, and the upstream `lfr/` Python
wrapper and `setup.py` (this repository wraps the binary itself, in
`_exp_revision/lfr_reference.py`).

## Patch: missing `return` statements

Four non-`void` functions fall off their end without returning, which is
undefined behaviour. At `-O0` it is harmless; at `-O1` and above g++ 11 turns
it into heap corruption and the program aborts or segfaults **after** writing
`network.dat` and `community.dat`, while writing the (unused) `statistics.dat`
histograms. Each got a `return 0;`, marked `// PATCHED` in place:

    src/print.cpp:12       int cherr()
    src/print.cpp:21       int cherr(double)
    src/histograms.cpp:643 int int_histogram(vector<int>&, ostream&)
    src/histograms.cpp:672 int int_histogram(deque<int>&, ostream&)

No generation logic is touched: the degree and community-size sequences, the
subgraph construction, `connect_all_the_parts` and the `erase_links` rewiring
are byte-for-byte upstream. The patch only lets the process exit cleanly so
the wrapper can trust its exit code.
