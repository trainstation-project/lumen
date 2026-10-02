// Launches the graph primitives' Metal kernels (lumen/ops/*/mps.metal,
// compiled as one library), and the fused kernels the MPS graph compiler
// generates (lumen/compiler/mps), into lumen's MPS stream. Generic: Rust
// (lumen/ops/*/mps.rs) names the kernel and passes its buffers and
// argument bytes, bound in that order.

#include "../stream/mps/mps.h"
#include "primitive_kernels.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/lock.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

id<MTLBuffer> lumen_mps_buffer(const void *ptr, size_t *offset);

// `source` compiled into a library on the stream's device, with IEEE
// semantics (NaN, infinities) rather than fast math, to match the reference
// executor (nil if that fails, with why in `error`).
static id<MTLLibrary> compile(NSString *source, NSError **error) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    if (queue == nil) {
        return nil;
    }
    MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    return [queue.device newLibraryWithSource:source options:options error:error];
}

// The kernels' library, compiled on first use (nil if that fails).
static id<MTLLibrary> library(void) {
    static id<MTLLibrary> lib = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSError *error = nil;
        lib = compile(kPrimitiveKernels, &error);
        if (lib == nil) {
            NSLog(@"lumen: compiling the MPS primitive kernels failed: %@", error);
        }
    });
    return lib;
}

// Pipelines by kernel name: the primitives' on first use, and those of
// lumen_mps_compile_kernels.
static os_unfair_lock cache_lock = OS_UNFAIR_LOCK_INIT;
static auto *cache = new std::unordered_map<std::string, id<MTLComputePipelineState>>();

// The pipeline for kernel `name`, created on first use.
static id<MTLComputePipelineState> pipeline(const char *name) {
    os_unfair_lock_lock(&cache_lock);
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
    os_unfair_lock_unlock(&cache_lock);
    return pso;
}

extern "C" {
// Compile Metal `source` (generated kernels, lumen/compiler/mps) and add a
// pipeline for each of its kernels, which lumen_mps_launch_kernel then
// launches by name. A kernel already added keeps its pipeline.
int lumen_mps_compile_kernels(const char *source) {
    @autoreleasepool {
        NSError *error = nil;
        id<MTLLibrary> lib = compile([NSString stringWithUTF8String:source], &error);
        if (lib == nil) {
            NSLog(@"lumen: compiling generated MPS kernels failed: %@", error);
            return -1;
        }
        for (NSString *name in lib.functionNames) {
            os_unfair_lock_lock(&cache_lock);
            bool known = cache->count(name.UTF8String) != 0;
            os_unfair_lock_unlock(&cache_lock);
            if (known) {
                continue;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            id<MTLComputePipelineState> pso = [lib.device newComputePipelineStateWithFunction:fn error:&error];
            if (pso == nil) {
                NSLog(@"lumen: creating the MPS pipeline %@ failed: %@", name, error);
                return -2;
            }
            os_unfair_lock_lock(&cache_lock);
            (*cache)[name.UTF8String] = pso;
            os_unfair_lock_unlock(&cache_lock);
        }
    }
    return 0;
}

int lumen_mps_launch_kernel(const char *name,
                            const void *const *buffers,
                            size_t nbuffers,
                            const void *const *args,
                            const size_t *arg_lens,
                            size_t nargs,
                            const size_t *grid,
                            int groups,
                            int timed,
                            lumen_mps_completion done,
                            void *context) {
    id<MTLComputePipelineState> pso = pipeline(name);
    for (int d = 0; d < 3; ++d) {
        if (grid[d] == 0 || grid[d] > UINT32_MAX) {
            return -2;
        }
    }
    if (pso == nil) {
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
        MTLSize size = MTLSizeMake(grid[0], grid[1], grid[2]);
        if (groups) {
            [encoder dispatchThreadgroups:size threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        } else {
            // Up to 256 threads a threadgroup, filling x first.
            NSUInteger total = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
            NSUInteger x = std::min<NSUInteger>(grid[0], total);
            NSUInteger y = std::min<NSUInteger>(grid[1], total / x);
            NSUInteger z = std::min<NSUInteger>(grid[2], total / (x * y));
            [encoder dispatchThreads:size threadsPerThreadgroup:MTLSizeMake(x, y, z)];
        }
        lumen_mps_stream_encoded(done, context);
    }
    return 0;
}
}
