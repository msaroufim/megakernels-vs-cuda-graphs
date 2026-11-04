"""Exercise actual scheduler wait control flow with scripted CPU counter loads."""

import shutil
import subprocess
from pathlib import Path

import pytest

KERNEL = Path(__file__).parents[1] / "deepspec/megakernel/csrc/dspark_v4_kernel.cu"


def test_waits_acquire_successful_counters(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = KERNEL.read_text()
    waits = source[
        source.index("__device__ __forceinline__ uint64_t wait_counter_at_least(") : source.index(
            "__device__ __forceinline__ void store_relaxed_u64("
        )
    ]
    readiness = source[
        source.index("__device__ __forceinline__ bool direct_dependencies_ready(") : source.index(
            "__device__ __forceinline__ ClaimedTask claim_ready_static_task("
        )
    ]
    program = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#define __device__
#define __forceinline__ inline
#define DSPARK_V4_DIRECT_DEPENDENCIES 2
std::vector<uint64_t> acquired_values, relaxed_values;
std::vector<const uint64_t*> acquired_addresses;
unsigned ai = 0, ri = 0;
bool pointer_mode = false;
uint64_t load_acquire_u64(const uint64_t* p) {
    acquired_addresses.push_back(p);
    return pointer_mode ? *p : acquired_values.at(ai++);
}
uint64_t load_relaxed_u64(const uint64_t* p) {
    return pointer_mode ? *p : relaxed_values.at(ri++);
}
void __nanosleep(unsigned) {}
uint64_t pack_epoch(uint32_t epoch, uint32_t value) {
    return (uint64_t(epoch) << 32) | value;
}
struct SchedulerWorkspace { uint64_t* publish_epoch; uint64_t* completed_tiles; };
constexpr int kDraftBlock = 1;
namespace generated {
constexpr int kPredecessorOffsets[] = {0, 0, 2};
constexpr int kPredecessors[] = {0, 2};
constexpr int kMarkovW2Phases[] = {2};
constexpr unsigned kWorkUnits[] = {3, 4, 5};
}
void script(std::vector<uint64_t> a, std::vector<uint64_t> r) {
    acquired_values = a; relaxed_values = r; ai = ri = 0;
    acquired_addresses.clear();
}
"""
        + waits
        + readiness
        + r"""
int main() {
    uint64_t counter = 0;
    script({7}, {});
    assert(wait_counter_exact(&counter, 7) == 7 && ai == 1 && ri == 0);
    script({0, 7}, {0, 7});
    assert(wait_counter_exact(&counter, 7) == 7 && ai == 2 && ri == 2);
    script({0, 8}, {7});
    assert(wait_counter_at_least(&counter, 7) == 8 && ai == 2 && ri == 1);
    script({0, 7}, {7});
    assert(wait_counter_at_least(&counter, 7) == 7 && ai == 2 && ri == 1);
    pointer_mode = true;
    uint64_t published[] = {0, pack_epoch(9, 1), pack_epoch(9, 2)};
    uint64_t completed[] = {pack_epoch(9, 3), 0, 0};
    SchedulerWorkspace workspace{published, completed};
    acquired_addresses.clear();
    int ready_cache = -1;
    assert(direct_dependencies_ready(workspace, 1, 9, ready_cache));
    for (const auto* required : {&published[1], &completed[0], &published[2]}) {
        assert(std::find(acquired_addresses.begin(), acquired_addresses.end(), required)
               != acquired_addresses.end());
    }
    completed[0] = pack_epoch(9, 2);
    ready_cache = -1;
    assert(!direct_dependencies_ready(workspace, 1, 9, ready_cache));
    assert(ready_cache == -1);
}
"""
    )
    cpp = tmp_path / "waits.cpp"
    executable = tmp_path / "waits"
    cpp.write_text(program)
    subprocess.run(
        [compiler, "-std=c++17", str(cpp), "-o", str(executable)], check=True, capture_output=True
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
