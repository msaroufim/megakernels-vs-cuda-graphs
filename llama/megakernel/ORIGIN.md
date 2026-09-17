# HazyResearch Llama megakernel

Initially copied byte-for-byte from [HazyResearch/Megakernels](https://github.com/HazyResearch/Megakernels/tree/7309cec801537b61fea3b50d7dfe454a6cde578e),
commit `7309cec801537b61fea3b50d7dfe454a6cde578e`.

| Here | Upstream |
|---|---|
| `Makefile`, `*.cu`, `*.cuh` | `demos/low-latency-llama/` |
| `include/` | `include/` |
| `LICENSE` | `LICENSE` (MIT, copyright HazyResearch) |

Edit these files directly. Both runners stage them into the pinned upstream
Python runtime and build for Hopper. ThunderKittens is fetched separately at
the commit in `../upstream/prepare.py`. Generated binaries stay outside this tree.
The published results describe the original source; subsequent edits need new measurements.
