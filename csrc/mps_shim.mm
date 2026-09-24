// C ABI over Metal for lumen's MPS allocator.
//
// Metal's API is Objective-C only, so — exactly like PyTorch's
// MPSAllocator.mm — the raw calls live in an Objective-C++ file that
// build.rs compiles on macOS. The Rust side (src/allocator/mps.rs) sees
// only the three extern "C" functions below.
//
// We allocate MTLBuffers in Shared storage mode: on Apple Silicon's unified
// memory, buffer.contents is an ordinary CPU pointer, which is what lets
// the Rust caching allocator treat Metal like any other raw-memory backend.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <unordered_map>

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

// contents pointer -> backing MTLBuffer, so lumen_mps_free can release the
// buffer given only the pointer the Rust side holds. Entries are __strong:
// inserting retains, erasing releases (ARC).
static std::unordered_map<void *, id<MTLBuffer> __strong> g_buffers;
static std::mutex g_buffers_mu;

extern "C" {
int lumen_mps_available(void) { return lumen_default_device() != nil ? 1 : 0; }

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
    g_buffers[contents] = buffer;
  }
  return contents;
}

void lumen_mps_free(void *contents) {
  if (contents == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_buffers_mu);
  g_buffers.erase(contents); // releases the MTLBuffer
}
}
