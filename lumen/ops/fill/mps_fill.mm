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
)";

struct Pipelines {
  id<MTLDevice> device = nil;
  id<MTLComputePipelineState> u16 = nil, u32 = nil, u64 = nil;
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

// Set count elements of elem_size bytes (1 to 8) at ptr, inside a lumen MPS
// segment, to the element at pattern: fillBuffer if its bytes are all the
// same, else the compute shader (2, 4 or 8 bytes). Submitted to the MPS
// stream without waiting; once the GPU finishes, done(context, GPU start,
// GPU end in host-clock seconds, ok) is called on a Metal thread.
//
// Returns 0 if submitted (done will be called), or without calling done:
// -1 if the memory is not mapped (e.g. a custom allocator's), -2 for an
// unsupported element size or no kernels.
int lumen_mps_fill(void *ptr, const void *pattern, size_t elem_size, size_t count,
                   lumen_mps_completion done, void *context) {
  const Pipelines &p = pipelines();
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
  const uint8_t *bytes = static_cast<const uint8_t *>(pattern);
  bool one_byte = std::all_of(bytes, bytes + elem_size, [&](uint8_t b) { return b == bytes[0]; });
  id<MTLComputePipelineState> pso = elem_size == 2 ? p.u16 : elem_size == 4 ? p.u32 : elem_size == 8 ? p.u64 : nil;
  if (queue == nil || count == 0 || (!one_byte && pso == nil)) {
    return -2;
  }
  // A no-copy view over the pages covering the elements.
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
  @autoreleasepool {
    id<MTLCommandBuffer> commands = lumen_mps_stream_begin();
    if (one_byte) {
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
