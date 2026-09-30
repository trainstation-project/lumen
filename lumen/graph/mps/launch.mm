// Launches the graph primitives' Metal kernels (kernels.metal) into lumen's
// MPS stream. Generic: Rust (lumen/graph/mps/mod.rs) names the kernel and
// passes its buffers and argument bytes, bound in that order.

#include "../../stream/mps/mps.h"
#include "primitive_kernels.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/lock.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset);

// The kernels' library, compiled on first use (nil if that fails).
static id<MTLLibrary> library(void) {
    static id<MTLLibrary> lib = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
        if (queue == nil) {
            return;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
        // IEEE semantics (NaN, infinities) rather than fast math, to match
        // the reference executor.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        NSError *error = nil;
        lib = [queue.device newLibraryWithSource:kPrimitiveKernels options:options error:&error];
        if (lib == nil) {
            NSLog(@"lumen: compiling the MPS primitive kernels failed: %@", error);
        }
    });
    return lib;
}

// The pipeline for kernel `name`, created on first use.
static id<MTLComputePipelineState> pipeline(const char *name) {
    static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
    static auto *cache = new std::unordered_map<std::string, id<MTLComputePipelineState>>();
    os_unfair_lock_lock(&lock);
    auto it = cache->find(name);
    id<MTLComputePipelineState> pso = it == cache->end() ? nil : it->second;
    if (pso == nil) {
        id<MTLLibrary> lib = library();
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (fn != nil) {
            pso = [lib.device newComputePipelineStateWithFunction:fn error:nil];
            (*cache)[name] = pso;
        }
    }
    os_unfair_lock_unlock(&lock);
    return pso;
}

extern "C" {
int lumen_mps_launch_kernel(const char *name,
                            const void *const *buffers,
                            size_t nbuffers,
                            const void *const *args,
                            const size_t *arg_lens,
                            size_t nargs,
                            size_t threads,
                            const size_t *groups,
                            const size_t *group,
                            int timed,
                            lumen_mps_completion done,
                            void *context) {
    id<MTLComputePipelineState> pso = pipeline(name);
    if (pso == nil || threads == 0 || threads > UINT32_MAX) {
        return -2;
    }
    @autoreleasepool {
        std::vector<id<MTLBuffer>> bound(nbuffers);
        std::vector<size_t> offsets(nbuffers, 0);
        for (size_t i = 0; i < nbuffers; ++i) {
            if (buffers[i] != nullptr) {
                bound[i] = lumen_mps_buffer(buffers[i], &offsets[i]);
                if (bound[i] == nil) {
                    return -1;
                }
            }
        }
        id<MTLComputeCommandEncoder> encoder = lumen_mps_stream_encoder(timed != 0);
        if (encoder == nil) {
            return -2;
        }
        [encoder setComputePipelineState:pso];
        for (size_t i = 0; i < nbuffers; ++i) {
            [encoder setBuffer:bound[i] offset:offsets[i] atIndex:i];
        }
        for (size_t j = 0; j < nargs; ++j) {
            [encoder setBytes:args[j] length:arg_lens[j] atIndex:nbuffers + j];
        }
        if (groups != nullptr) {
            [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
                    threadsPerThreadgroup:MTLSizeMake(group[0], group[1], group[2])];
        } else {
            NSUInteger width = std::min<NSUInteger>({pso.maxTotalThreadsPerThreadgroup, 256, threads});
            [encoder dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        }
        lumen_mps_stream_encoded(done, context);
    }
    return 0;
}
}
