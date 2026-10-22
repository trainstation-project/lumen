#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/vm_page_size.h>

#include <map>
#include <mutex>

static id<MTLDevice> lumen_default_device(void) {
    static id<MTLDevice> device = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        device = MTLCreateSystemDefaultDevice();
    });
    return device;
}

static std::map<char *, id<MTLBuffer> __strong> g_buffers;
static std::map<char *, id<MTLBuffer> __strong> g_heaps;
static std::mutex g_buffers_mu;

static id<MTLBuffer> containing(const std::map<char *, id<MTLBuffer> __strong> &buffers, char *p, size_t *offset) {
    auto it = buffers.upper_bound(p);
    if (it == buffers.begin()) {
        return nil;
    }
    --it;
    if (p >= it->first + it->second.length) {
        return nil;
    }
    *offset = static_cast<size_t>(p - it->first);
    return it->second;
}

id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset) {
    char *p = static_cast<char *>(const_cast<void *>(ptr));
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    id<MTLBuffer> buffer = containing(g_buffers, p, offset);
    return buffer != nil ? buffer : containing(g_heaps, p, offset);
}

extern "C" {
int lumen_mps_available(void) { return lumen_default_device() != nil ? 1 : 0; }

void *lumen_mps_buffer_pointer(const void *ptr, size_t *offset) {
    return (__bridge void *)lumen_mps_buffer(ptr, offset);
}

typedef struct {
    size_t alignment;
    size_t page_size;
    size_t max_buffer_length;
    size_t recommended_max_working_set;
} lumen_mps_limits_t;

int lumen_mps_limits(lumen_mps_limits_t *out) {
    id<MTLDevice> device = lumen_default_device();
    if (device == nil) {
        return 0;
    }
    MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    out->alignment = [device heapBufferSizeAndAlignWithLength:1 options:options].align;
    out->page_size = vm_page_size;
    out->max_buffer_length = device.maxBufferLength;
    out->recommended_max_working_set = (size_t)device.recommendedMaxWorkingSetSize;
    return 1;
}

void *lumen_mps_alloc(size_t nbytes) {
    id<MTLDevice> device = lumen_default_device();
    if (device == nil) {
        return nullptr;
    }
    id<MTLBuffer> buffer =
        [device newBufferWithLength:nbytes
                            options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
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
    g_buffers.erase(static_cast<char *>(contents));
    g_heaps.erase(static_cast<char *>(contents));
}

static constexpr MTLResourceOptions kPlaced = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;

void *lumen_mps_alloc_heap(size_t nbytes) {
    id<MTLDevice> device = lumen_default_device();
    if (device == nil) {
        return nullptr;
    }
    MTLHeapDescriptor *d = [MTLHeapDescriptor new];
    d.type = MTLHeapTypePlacement;
    d.storageMode = MTLStorageModeShared;
    d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    d.size = nbytes;
    id<MTLHeap> heap = [device newHeapWithDescriptor:d];
    id<MTLBuffer> whole = [heap newBufferWithLength:nbytes options:kPlaced offset:0];
    if (whole == nil) {
        return nullptr;
    }
    void *contents = whole.contents;
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    g_heaps[static_cast<char *>(contents)] = whole;
    return contents;
}

int lumen_mps_place(void *ptr, size_t nbytes) {
    char *p = static_cast<char *>(ptr);
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    size_t offset = 0;
    id<MTLBuffer> whole = containing(g_heaps, p, &offset);
    if (whole == nil) {
        return -1;
    }
    id<MTLBuffer> placed = [whole.heap newBufferWithLength:nbytes options:kPlaced offset:whole.heapOffset + offset];
    if (placed == nil) {
        return -1;
    }
    g_buffers[p] = placed;
    return 0;
}

void lumen_mps_unplace(void *ptr) {
    std::lock_guard<std::mutex> lock(g_buffers_mu);
    g_buffers.erase(static_cast<char *>(ptr));
}
}
