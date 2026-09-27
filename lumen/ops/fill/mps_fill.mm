// The MPS fill_ kernel's GPU side: a Metal compute shader writing 2-, 4- or
// 8-byte elements (PyTorch's MPS fill_ is a compute kernel too; Metal's blit
// fillBuffer only sets bytes). The shader is compiled from source at run
// time, so no offline Metal compiler is needed. build.rs compiles this file;
// ops/fill/mps.rs calls lumen_mps_fill.
//
// The kernel binds a no-copy MTLBuffer view over the tensor's pages rather
// than the allocator's buffer: lumen's MPS segments are page-aligned and a
// whole number of pages, so rounding the range out to pages stays inside
// its segment.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_page_size.h>

#include <algorithm>

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
)";

struct Pipelines {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> u16 = nil, u32 = nil, u64 = nil;
};

// Compiled once, on first use.
static const Pipelines &pipelines(void) {
  static Pipelines p;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      return;
    }
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
    p.queue = [device newCommandQueue];
    p.u16 = pipeline(@"fill_u16");
    p.u32 = pipeline(@"fill_u32");
    p.u64 = pipeline(@"fill_u64");
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
// Set count elements of elem_size bytes (2, 4 or 8) at ptr, inside a lumen
// MPS segment, to the element at pattern; wait for the GPU. Its start/end
// times (host-clock seconds) go to gpu_start and gpu_end, which may be null.
// Returns 0 on success, -1 if the memory is not mapped (e.g. a custom
// allocator's), -2 for an unsupported element size or no kernels, -3 if
// the GPU work failed.
int lumen_mps_fill(void *ptr, const void *pattern, size_t elem_size, size_t count,
                   double *gpu_start, double *gpu_end) {
  if (count == 0) {
    return 0;
  }
  const Pipelines &p = pipelines();
  id<MTLComputePipelineState> pso = elem_size == 2 ? p.u16 : elem_size == 4 ? p.u32 : elem_size == 8 ? p.u64 : nil;
  if (pso == nil || p.queue == nil) {
    return -2;
  }
  // The pages covering the elements.
  uintptr_t first = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t pages = first & ~(uintptr_t)(vm_page_size - 1);
  uintptr_t end = first + elem_size * count;
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
  uint64_t start = (first - pages) / elem_size;
  @autoreleasepool {
    id<MTLCommandBuffer> commands = [p.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:pattern length:elem_size atIndex:1];
    [encoder setBytes:&start length:sizeof(start) atIndex:2];
    NSUInteger width = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, count);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (gpu_start != nullptr) {
      *gpu_start = commands.GPUStartTime;
    }
    if (gpu_end != nullptr) {
      *gpu_end = commands.GPUEndTime;
    }
    return commands.status == MTLCommandBufferStatusCompleted ? 0 : -3;
  }
}
}
