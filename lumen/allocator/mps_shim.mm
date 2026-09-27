// C ABI over Metal for lumen's MPS allocator.
//
// Metal's API is Objective-C only, so — exactly like PyTorch's
// MPSAllocator.mm — the raw calls live in an Objective-C++ file that
// build.rs compiles on macOS. The Rust side (mps.rs, next to this file) sees
// only the extern "C" functions below.
//
// We allocate MTLBuffers in Shared storage mode: on Apple Silicon's unified
// memory, buffer.contents is an ordinary CPU pointer, which is what lets
// the Rust caching allocator treat Metal like any other raw-memory backend.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <mach/vm_page_size.h>

#include <map>
#include <mutex>

// The system default device, created lazily (MTLCreateSystemDefaultDevice is
// documented as expensive; call it once).
static id<MTLDevice> lumen_default_device(void) {
  static id<MTLDevice> device = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
  });
  return device;
}

// A command queue for blit work (fills), created with the device.
static id<MTLCommandQueue> lumen_command_queue(void) {
  static id<MTLCommandQueue> queue = nil;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    queue = [lumen_default_device() newCommandQueue];
  });
  return queue;
}

// contents pointer -> backing MTLBuffer, so lumen_mps_free can release the
// buffer given only the pointer the Rust side holds. Entries are __strong:
// inserting retains, erasing releases (ARC). Ordered, so a pointer *inside*
// a buffer (a caching-allocator block within a segment) finds its buffer.
static std::map<char *, id<MTLBuffer> __strong> g_buffers;
static std::mutex g_buffers_mu;

extern "C" {
int lumen_mps_available(void) { return lumen_default_device() != nil ? 1 : 0; }

// Device properties the allocator's size math needs, mirroring what
// MPSHeapAllocatorImpl reads from Metal. Returns 0 if there is no device.
typedef struct {
  size_t alignment;                  // heap buffer placement alignment
  size_t page_size;                  // vm_page_size
  size_t max_buffer_length;          // MTLDevice.maxBufferLength
  size_t recommended_max_working_set; // MTLDevice.recommendedMaxWorkingSetSize
} lumen_mps_limits_t;

int lumen_mps_limits(lumen_mps_limits_t *out) {
  id<MTLDevice> device = lumen_default_device();
  if (device == nil) {
    return 0;
  }
  // Same options as the buffers lumen_mps_alloc creates (and as MPS's shared
  // pools use); MPS's BufferPool queries the alignment with length 1.
  MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
  out->alignment = [device heapBufferSizeAndAlignWithLength:1 options:options].align;
  out->page_size = vm_page_size;
  out->max_buffer_length = device.maxBufferLength;
  out->recommended_max_working_set = (size_t)device.recommendedMaxWorkingSetSize;
  return 1;
}

// Returns the contents pointer of a new Shared-mode MTLBuffer, or nullptr.
void *lumen_mps_alloc(size_t nbytes) {
  id<MTLDevice> device = lumen_default_device();
  if (device == nil) {
    return nullptr;
  }
  id<MTLBuffer> buffer = [device newBufferWithLength:nbytes
                                             options:MTLResourceStorageModeShared];
  if (buffer == nil) {
    return nullptr;
  }
  void *contents = buffer.contents;
  {
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    g_buffers[static_cast<char *>(contents)] = buffer;
  }
  return contents;
}

void lumen_mps_free(void *contents) {
  if (contents == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_buffers_mu);
  g_buffers.erase(static_cast<char *>(contents)); // releases the MTLBuffer
}

// The host clock Metal's GPUStartTime/GPUEndTime are in (mach_absolute_time,
// as seconds), so the profiler can map GPU timestamps onto its own clock.
double lumen_mps_host_time(void) {
  static mach_timebase_info_data_t timebase;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    mach_timebase_info(&timebase);
  });
  return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
}

// Set nbytes at ptr (anywhere inside a buffer from lumen_mps_alloc) to value,
// with a blit encoder's fillBuffer — Metal's memset — and wait for it. The
// command buffer's GPU start/end times (host-clock seconds) go to gpu_start
// and gpu_end, which may be null. Returns 0 on success, -1 if
// ptr..ptr+nbytes is not inside one buffer.
int lumen_mps_memset(void *ptr, uint8_t value, size_t nbytes, double *gpu_start, double *gpu_end) {
  if (nbytes == 0) {
    return 0;
  }
  char *p = static_cast<char *>(ptr);
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0;
  {
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    auto it = g_buffers.upper_bound(p); // first buffer starting after p
    if (it == g_buffers.begin()) {
      return -1;
    }
    --it;
    offset = static_cast<NSUInteger>(p - it->first);
    if (offset + nbytes > it->second.length) {
      return -1;
    }
    buffer = it->second; // retained for the blit below
  }
  @autoreleasepool {
    id<MTLCommandBuffer> commands = [lumen_command_queue() commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit fillBuffer:buffer range:NSMakeRange(offset, nbytes) value:value];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (gpu_start != nullptr) {
      *gpu_start = commands.GPUStartTime;
    }
    if (gpu_end != nullptr) {
      *gpu_end = commands.GPUEndTime;
    }
    return commands.status == MTLCommandBufferStatusCompleted ? 0 : -1;
  }
}
}
