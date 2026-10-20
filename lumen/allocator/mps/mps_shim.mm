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
static std::mutex g_buffers_mu;

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
    g_buffers.erase(static_cast<char *>(contents));
}
}
