#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

#include "../../stream/mps.h"

// The fill kernels live in fill.metal; build.rs embeds its text here as the
// C++ raw string `kMpsFillSource` (compiled at runtime by Metal below).
#include "mps_fill_source.h"

// In allocator/mps_shim.mm.
id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset);

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
    id<MTLLibrary> library = [device newLibraryWithSource:kMpsFillSource options:nil error:&error];
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
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:pattern length:elem_size atIndex:1];
    [encoder setBytes:&start length:sizeof(start) atIndex:2];
    if (strided) {
      // 2D dispatch (PyTorch: fill_mps_kernel's strided branch): y indexes
      // dim 0, x walks the inner dims, so threads consecutive in x write
      // consecutive addresses in the innermost dimension.
      uint32_t dims = static_cast<uint32_t>(ndim);
      [encoder setBytes:sizes length:ndim * sizeof(size_t) atIndex:3];
      [encoder setBytes:strides length:ndim * sizeof(size_t) atIndex:4];
      [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
      NSUInteger dim0 = sizes[0];
      NSUInteger inner = count / dim0;
      NSUInteger x = std::min<NSUInteger>(std::max<NSUInteger>(inner, 1), width);
      [encoder dispatchThreads:MTLSizeMake(inner, dim0, 1)
          threadsPerThreadgroup:MTLSizeMake(x, 1, 1)];
    } else {
      // The contiguous kernels need the element count for the tail loop,
      // and loop with a grid stride over vectorized stores, so the grid is
      // a few waves rather than one thread per element.
      [encoder setBytes:&n length:sizeof(n) atIndex:3];
      NSUInteger waves = 32;
      NSUInteger grid_threads = std::min<NSUInteger>(width * waves, ((count + width - 1) / width) * width);
      grid_threads = std::max<NSUInteger>(grid_threads, width);
      [encoder dispatchThreads:MTLSizeMake(grid_threads, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    lumen_mps_stream_encoded(done, context);
  }
  return 0;
}
}
