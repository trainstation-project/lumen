#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_page_size.h>

#include <algorithm>

// In stream/mps.mm.
extern "C" void *lumen_mps_stream_queue(void);
id<MTLCommandBuffer> lumen_mps_stream_begin(void);
void lumen_mps_stream_commit(id<MTLCommandBuffer> commands);

static NSString *const kSource = @R"(
#include <metal_stdlib>
using namespace metal;

// buf[start + i] = value for every thread i. The buffer is bound at offset 0
// and the start passed in elements, so element-aligned starts need no
// binding alignment.
#define FILL(NAME, T)                                                  \
  kernel void NAME(device T *buf [[buffer(0)]],                        \
                   constant T &value [[buffer(1)]],                    \
                   constant ulong &start [[buffer(2)]],                \
                   uint i [[thread_position_in_grid]]) {               \
    buf[start + i] = value;                                            \
  }
FILL(fill_u16, ushort)
FILL(fill_u32, uint)
FILL(fill_u64, ulong)

// A strided view (PyTorch: fill_scalar_strided): thread i writes the view's
// i-th element in row-major order, at start + sum(index[d] * strides[d]).
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
  id<MTLComputePipelineState> u16 = nil, u32 = nil, u64 = nil;
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

// Whether [start, end) lies in one mapped region of this process.
static bool mapped(mach_vm_address_t start, mach_vm_address_t end) {
  mach_vm_address_t region = start;
  mach_vm_size_t size = 0;
  vm_region_basic_info_data_64_t info;
  mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object = MACH_PORT_NULL;
  kern_return_t kr = mach_vm_region(mach_task_self(), &region, &size, VM_REGION_BASIC_INFO_64,
                                    (vm_region_info_t)&info, &count, &object);
  return kr == KERN_SUCCESS && region <= start && end <= region + size;
}

extern "C" {
typedef void (*lumen_mps_completion)(void *context, double gpu_start, double gpu_end, int ok);

// Set the count elements of elem_size bytes (1, 2, 4 or 8) starting at ptr,
// inside a lumen MPS segment, to the element at pattern. With strides null
// they are contiguous: fillBuffer if the pattern's bytes are all the same,
// else the dense compute shader. Otherwise they are a view of ndim sizes and
// strides (in elements): the strided compute shader. Submitted to the MPS
// stream without waiting; once the GPU finishes, done(context, GPU start,
// GPU end in host-clock seconds, ok) is called on a Metal thread.
//
// Returns 0 if submitted (done will be called), or without calling done:
// -1 if the memory is not mapped (e.g. a custom allocator's), -2 for an
// unsupported element size, too many elements or no kernels.
int lumen_mps_fill(void *ptr, const void *pattern, size_t elem_size, size_t count,
                   const size_t *sizes, const size_t *strides, size_t ndim,
                   lumen_mps_completion done, void *context) {
  const Pipelines &p = pipelines();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
  const uint8_t *bytes = static_cast<const uint8_t *>(pattern);
  bool strided = strides != nullptr;
  bool blit = !strided && std::all_of(bytes, bytes + elem_size, [&](uint8_t b) { return b == bytes[0]; });
  id<MTLComputePipelineState> pso =
      strided ? (elem_size == 1 ? p.s8 : elem_size == 2 ? p.s16 : elem_size == 4 ? p.s32 : elem_size == 8 ? p.s64 : nil)
              : (elem_size == 2 ? p.u16 : elem_size == 4 ? p.u32 : elem_size == 8 ? p.u64 : nil);
  // Thread indices are 32-bit.
  if (queue == nil || count == 0 || count > UINT32_MAX || (!blit && pso == nil)) {
    return -2;
  }
  // The elements span 1 + sum((sizes[d] - 1) * strides[d]) elements.
  size_t span = count;
  if (strided) {
    span = 1;
    for (size_t d = 0; d < ndim; d++) {
      span += (sizes[d] - 1) * strides[d];
    }
  }
  // A no-copy view over the pages covering the elements.
  uintptr_t first = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t pages = first & ~(uintptr_t)(vm_page_size - 1);
  uintptr_t end = first + elem_size * span;
  size_t length = ((end - pages) + vm_page_size - 1) & ~(size_t)(vm_page_size - 1);
  if (!mapped(pages, pages + length)) {
    return -1;
  }
  id<MTLBuffer> buffer = [p.device newBufferWithBytesNoCopy:reinterpret_cast<void *>(pages)
                                                     length:length
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
  if (buffer == nil) {
    return -1;
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commands = lumen_mps_stream_begin();
    if (blit) {
      id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
      [blit fillBuffer:buffer range:NSMakeRange(first - pages, elem_size * count) value:bytes[0]];
      [blit endEncoding];
    } else {
      uint64_t start = (first - pages) / elem_size;
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      [encoder setComputePipelineState:pso];
      [encoder setBuffer:buffer offset:0 atIndex:0];
      [encoder setBytes:pattern length:elem_size atIndex:1];
      [encoder setBytes:&start length:sizeof(start) atIndex:2];
      if (strided) {
        uint32_t dims = static_cast<uint32_t>(ndim);
        [encoder setBytes:sizes length:ndim * sizeof(size_t) atIndex:3];
        [encoder setBytes:strides length:ndim * sizeof(size_t) atIndex:4];
        [encoder setBytes:&dims length:sizeof(dims) atIndex:5];
      }
      NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, count);
      [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
      [encoder endEncoding];
    }
    [commands addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      done(context, cb.GPUStartTime, cb.GPUEndTime, cb.status == MTLCommandBufferStatusCompleted);
    }];
    lumen_mps_stream_commit(commands);
  }
  return 0;
}
}
