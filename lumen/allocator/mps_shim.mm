// C ABI over Metal for lumen's MPS allocator.
//
// Metal's API is Objective-C only, so — exactly like PyTorch's
// MPSAllocator.mm — the raw calls live in an Objective-C++ file that
// build.rs compiles on macOS. The Rust side (mps.rs, next to this file) sees
// only the extern "C" functions below.
//
// We allocate MTLBuffers in Shared storage mode: on Apple Silicon's unified
// memory, buffer.contents is an ordinary CPU pointer, which is what lets
// the Rust static allocator treat Metal like any other raw-memory backend.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

// contents pointer -> backing MTLBuffer, so lumen_mps_free can release the
// buffer given only the pointer the Rust side holds. Entries are __strong:
// inserting retains, erasing releases (ARC). Ordered, so a pointer *inside*
// a buffer (an allocation within the static allocator's region) finds its buffer.
static std::map<char *, id<MTLBuffer> __strong> g_buffers;
static std::mutex g_buffers_mu;

// The MTLBuffer of the segment holding ptr, and ptr's byte offset in it, or
// nil if ptr is not in one (for ops to bind tensor memory, like PyTorch's
// getMTLBufferStorage).
id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset) {
    char *p = static_cast<char *>(const_cast<void *>(ptr));
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    auto it = g_buffers.upper_bound(p);
    if (it == g_buffers.begin()) {
        return nil;
    }
    --it;
    if (p >= it->first + it->second.length) {
        return nil;
    }
    *offset = static_cast<size_t>(p - it->first);
    return it->second;
}

extern "C" {
int lumen_mps_available(void) { return lumen_default_device() != nil ? 1 : 0; }

// Device properties the allocator's size math needs, mirroring what
// MPSHeapAllocatorImpl reads from Metal. Returns 0 if there is no device.
typedef struct {
    size_t alignment;                   // heap buffer placement alignment
    size_t page_size;                   // vm_page_size
    size_t max_buffer_length;           // MTLDevice.maxBufferLength
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
    id<MTLBuffer> buffer = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
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
}
