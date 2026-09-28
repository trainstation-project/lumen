#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

#include "../../stream/mps.h"

// In allocator/mps_shim.mm.
id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset);

static NSString *const kSource = @R"(
#include <metal_stdlib>
using namespace metal;

#define FILL(NAME, T, V)                                               \
  kernel void NAME(device T *buf [[buffer(0)]],                        \
                   constant T &value [[buffer(1)]],                    \
                   constant ulong &start [[buffer(2)]],                \
                   constant ulong &count [[buffer(3)]],                \
                   uint i [[thread_position_in_grid]],                 \
                   uint nthreads [[threads_per_grid]]) {               \
    constexpr uint vec = sizeof(V) / sizeof(T);                        \
    const ulong vec_count = count / vec;                              \
    device V *wide = reinterpret_cast<device V *>(buf + start);        \
    V pattern = V(value);                                              \
    for (ulong j = i; j < vec_count; j += nthreads) {                  \
      wide[j] = pattern;                                              \
    }                                                                  \
    if (i == 0) {                                                      \
      for (ulong j = vec_count * vec; j < count; ++j) {                \
        buf[start + j] = value;                                        \
      }                                                                \
    }                                                                  \
  }
FILL(fill_u8, uchar, uchar4)
FILL(fill_u16, ushort, ushort4)
FILL(fill_u32, uint, uint4)
FILL(fill_u64, ulong, ulong2)

#define FILL_STRIDED(NAME, T)                                          \
  kernel void NAME(device T *buf [[buffer(0)]],                        \
                   constant T &value [[buffer(1)]],                    \
                   constant ulong &start [[buffer(2)]],                \
                   constant ulong *sizes [[buffer(3)]],                \
                   constant ulong *strides [[buffer(4)]],              \
                   constant uint &ndim [[buffer(5)]],                  \
                   uint i [[thread_position_in_grid]]) {               \
    ulong rest = i, offset = 0;                                        \
    for (uint d = ndim; d-- > 0;) {                                    \
      offset += (rest % sizes[d]) * strides[d];                        \
      rest /= sizes[d];                                                \
    }                                                                  \
    buf[start + offset] = value;                                       \
  }
FILL_STRIDED(fill_strided_u8, uchar)
FILL_STRIDED(fill_strided_u16, ushort)
FILL_STRIDED(fill_strided_u32, uint)
FILL_STRIDED(fill_strided_u64, ulong)
)";

struct Pipelines {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> u8 = nil, u16 = nil, u32 = nil, u64 = nil;
  id<MTLComputePipelineState> s8 = nil, s16 = nil, s32 = nil, s64 = nil; // strided
};

// Compiled once, on first use.
static const Pipelines &pipelines(void) {
  static Pipelines p;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    if (queue == nil) {
      return;
    }
    id<MTLDevice> device = queue.device;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (library == nil) {
      NSLog(@"lumen: compiling the MPS fill kernels failed: %@", error);
      return;
    }
    auto pipeline = [&](NSString *name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [library newFunctionWithName:name];
      return fn == nil ? nil : [device newComputePipelineStateWithFunction:fn error:nil];
    };
    p.device = device;
    p.u8 = pipeline(@"fill_u8");
    p.u16 = pipeline(@"fill_u16");
    p.u32 = pipeline(@"fill_u32");
    p.u64 = pipeline(@"fill_u64");
    p.s8 = pipeline(@"fill_strided_u8");
    p.s16 = pipeline(@"fill_strided_u16");
    p.s32 = pipeline(@"fill_strided_u32");
    p.s64 = pipeline(@"fill_strided_u64");
  });
  return p;
}

extern "C" {
// Set the count elements of elem_size bytes (1, 2, 4 or 8) starting at ptr,
// inside a lumen MPS segment, to the element at pattern. With strides null
// they are contiguous; otherwise a view of ndim sizes and strides (in
// elements). Encoded into the MPS stream without waiting (timed: sampled for
// the profiler); once the GPU has run it, done(context, GPU start, GPU end
// in host-clock seconds, ok) is called on a Metal thread.
//
// Returns 0 if encoded (done will be called), or without calling done:
// -1 if ptr is not in a lumen MPS segment (e.g. a custom allocator's), -2
// for an unsupported element size, too many elements or no kernels.
int lumen_mps_fill(void *ptr, const void *pattern, size_t elem_size, size_t count,
                   const size_t *sizes, const size_t *strides, size_t ndim, int timed,
                   lumen_mps_completion done, void *context) {
  const Pipelines &p = pipelines();
  bool strided = strides != nullptr;
  auto pick = [&](id<MTLComputePipelineState> b1, id<MTLComputePipelineState> b2,
                  id<MTLComputePipelineState> b4, id<MTLComputePipelineState> b8) {
    return elem_size == 1 ? b1 : elem_size == 2 ? b2 : elem_size == 4 ? b4 : elem_size == 8 ? b8 : nil;
  };
  id<MTLComputePipelineState> pso = strided ? pick(p.s8, p.s16, p.s32, p.s64) : pick(p.u8, p.u16, p.u32, p.u64);
  // Thread indices are 32-bit; the strided kernels index one element per
  // thread, so they cap out there. Contiguous kernels loop with a grid
  // stride and only need the count to fit in 64 bits.
  if (pso == nil || count == 0 || (strided && count > UINT32_MAX)) {
    return -2;
  }
  size_t offset = 0;
  id<MTLBuffer> buffer = lumen_mps_buffer(ptr, &offset);
  if (buffer == nil) {
    return -1;
  }
  @autoreleasepool {
    id<MTLComputeCommandEncoder> encoder = lumen_mps_stream_encoder(timed != 0);
    if (encoder == nil) {
      return -2;
    }
    // Bound at offset 0, with the start in elements (blocks are element
    // aligned in their segment).
    uint64_t start = offset / elem_size;
    uint64_t n = count;
    NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
    // Contiguous kernels loop with a grid stride over vectorized stores, so
    // the grid is sized to a few waves rather than one thread per element;
    // strided kernels do one element per thread and cover their grid exactly.
    NSUInteger grid_threads;
    if (strided) {
      grid_threads = ((count + width - 1) / width) * width;
    } else {
      NSUInteger waves = 32;
      grid_threads = std::min<NSUInteger>(width * waves, ((count + width - 1) / width) * width);
      grid_threads = std::max<NSUInteger>(grid_threads, width);
    }
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:pattern length:elem_size atIndex:1];
    [encoder setBytes:&start length:sizeof(start) atIndex:2];
    if (strided) {
      uint32_t dims = static_cast<uint32_t>(ndim);
      [encoder setBytes:sizes length:ndim * sizeof(size_t) atIndex:3];
      [encoder setBytes:strides length:ndim * sizeof(size_t) atIndex:4];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
    } else {
      // The contiguous kernels need the element count for the tail loop.
      [encoder setBytes:&n length:sizeof(n) atIndex:3];
    }
    [encoder dispatchThreads:MTLSizeMake(grid_threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    lumen_mps_stream_encoded(done, context);
  }
  return 0;
}
}
