#pragma once

#include <cute/arch/tmem_allocator_sm100.hpp>

namespace dspark_tmem {

#ifdef DSPARK_V4_PERSISTENT_TMEM
// One CTA owns one reservation for the entire GPU-scheduled proposal. Every
// phase drains its asynchronous TMEM users and reaches the existing CTA
// barrier before the next phase can reuse the same addresses.
constexpr int kReservedColumns = 128;

__device__ __forceinline__ uint32_t& base_slot() {
  __shared__ uint32_t base;
  return base;
}

__device__ __forceinline__ void reserve() {
  if (threadIdx.x < 32) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(kReservedColumns, &base_slot());
  }
  __syncthreads();
}

__device__ __forceinline__ void release() {
  __syncthreads();
  if (threadIdx.x < 32) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.free(base_slot(), kReservedColumns);
  }
}

class Allocator1Sm {
 public:
  __device__ __forceinline__ void allocate(int columns, uint32_t* destination) {
    if (columns > kReservedColumns) {
      __trap();
    }
    if ((threadIdx.x & 31) == 0) {
      *destination = base_slot();
    }
  }

  __device__ __forceinline__ void free(uint32_t, int) {}
};
#else
using Allocator1Sm = cute::TMEM::Allocator1Sm;
#endif

// Routed GEMMs can claim several ranges during one proposal. Keep their
// reservation distinct from every other phase's TMEM, and reuse only this
// routed allocation. The existing body barriers drain users between claims.
struct RoutedReservation {
  uint32_t base;
  int columns;
};

__device__ __forceinline__ RoutedReservation& routed_reservation() {
  __shared__ RoutedReservation reservation;
  return reservation;
}

__device__ __forceinline__ void initialize_routed_reservation() {
  if (threadIdx.x == 0) routed_reservation().columns = 0;
  __syncthreads();
}

class RoutedAllocator1Sm {
 public:
  // Exactly the same warp must issue all allocations in a CTA.
  __device__ __forceinline__ void allocate(int columns, uint32_t* destination) {
    auto& reservation = routed_reservation();
    if (reservation.columns == 0) {
      cute::TMEM::Allocator1Sm{}.allocate(columns, destination);
    } else {
      if (columns > reservation.columns) __trap();
      if ((threadIdx.x & 31) == 0) *destination = reservation.base;
    }
  }
  __device__ __forceinline__ void free(uint32_t, int) {}
};

// Called after the allocation's existing CTA barrier, once its shared result
// is available. A later claim reads these fields only after the closing
// barrier.
__device__ __forceinline__ void remember_routed_reservation(uint32_t base,
                                                            int columns) {
  if (threadIdx.x == 0 && routed_reservation().columns == 0)
    routed_reservation() = {base, columns};
}

__device__ __forceinline__ void release_routed_reservation() {
  __syncthreads();
  if (threadIdx.x < 32 && routed_reservation().columns != 0)
    cute::TMEM::Allocator1Sm{}.free(routed_reservation().base,
                                    routed_reservation().columns);
}

}  // namespace dspark_tmem
