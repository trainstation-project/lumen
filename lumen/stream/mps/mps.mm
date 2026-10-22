#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <os/lock.h>

#include <vector>

#include "mps.h"

extern "C" {
void *lumen_mps_stream_queue(void) {
    static id<MTLCommandQueue> queue = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        queue = [MTLCreateSystemDefaultDevice() newCommandQueue];
    });
    return (__bridge void *)queue;
}
}

static constexpr size_t kOpsPerCommit = 32;

struct Op {
    lumen_mps_completion done;
    void *context;
    long sample;
};

static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
static id<MTLCommandBuffer> commands = nil;
static id<MTLComputeCommandEncoder> encoder = nil;
static bool encoder_timed = false;
static id<MTLCounterSampleBuffer> samples = nil;
static std::vector<Op> ops;
static uint64_t encoded_total = 0;

static void end_encoder(void) {
    [encoder endEncoding];
    encoder = nil;
}

static constexpr long kSampleBuffers = 24;
static os_unfair_lock pool_lock = OS_UNFAIR_LOCK_INIT;
static NSMutableArray<id<MTLCounterSampleBuffer>> *free_samples = nil;
static dispatch_semaphore_t sample_slots = nil;

static id<MTLCounterSampleBuffer> create_samples(id<MTLDevice> device) {
    if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        return nil;
    }
    for (id<MTLCounterSet> set in device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
            d.counterSet = set;
            d.storageMode = MTLStorageModeShared;
            d.sampleCount = 2 * kOpsPerCommit;
            return [device newCounterSampleBufferWithDescriptor:d error:nil];
        }
    }
    return nil;
}

static id<MTLCounterSampleBuffer> new_samples(id<MTLDevice> device) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        free_samples = [NSMutableArray array];
        sample_slots = dispatch_semaphore_create(kSampleBuffers);
    });
    dispatch_semaphore_wait(sample_slots, DISPATCH_TIME_FOREVER);
    os_unfair_lock_lock(&pool_lock);
    id<MTLCounterSampleBuffer> samples = free_samples.lastObject;
    if (samples != nil) {
        [free_samples removeLastObject];
    }
    os_unfair_lock_unlock(&pool_lock);
    if (samples == nil) {
        samples = create_samples(device);
    }
    if (samples == nil) {
        dispatch_semaphore_signal(sample_slots);
    }
    return samples;
}

static void return_samples(id<MTLCounterSampleBuffer> samples) {
    os_unfair_lock_lock(&pool_lock);
    [free_samples addObject:samples];
    os_unfair_lock_unlock(&pool_lock);
    dispatch_semaphore_signal(sample_slots);
}

static void commit(void) {
    if (commands == nil) {
        return;
    }
    if (encoder != nil) {
        end_encoder();
    }
    id<MTLCounterSampleBuffer> timestamps = samples;
    std::vector<Op> committed = std::move(ops);
    ops.clear();
    MTLTimestamp cpu = 0, gpu = 0;
    if (timestamps != nil) {
        [commands.device sampleTimestamps:&cpu gpuTimestamp:&gpu];
    }
    [commands addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        int ok = cb.status == MTLCommandBufferStatusCompleted;
        NSData *data = nil;
        if (timestamps != nil && ok) {
            data = [timestamps resolveCounterRange:NSMakeRange(0, 2 * kOpsPerCommit)];
        }
        auto *results = static_cast<const MTLCounterResultTimestamp *>(data.bytes);
        auto seconds = [&](uint64_t t) { return ((double)cpu + ((double)t - (double)gpu)) / 1e9; };
        for (const Op &op : committed) {
            double start = cb.GPUStartTime, end = cb.GPUEndTime;
            if (results != nullptr && op.sample >= 0) {
                uint64_t s = results[op.sample].timestamp, e = results[op.sample + 1].timestamp;
                if (s != MTLCounterErrorValue && e != MTLCounterErrorValue && s != 0 && e >= s) {
                    start = seconds(s);
                    end = seconds(e);
                }
            }
            op.done(op.context, start, end, ok);
        }
        if (timestamps != nil) {
            return_samples(timestamps);
        }
    }];
    [commands commit];
    commands = nil;
    samples = nil;
}

id<MTLComputeCommandEncoder> lumen_mps_stream_encoder(bool timed) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    if (queue == nil) {
        return nil;
    }
    os_unfair_lock_lock(&lock);
    if (commands == nil) {
        commands = [queue commandBuffer];
    }
    if (timed && samples == nil) {
        samples = new_samples(queue.device);
    }
    timed = timed && samples != nil;
    if (encoder != nil && (timed || encoder_timed)) {
        end_encoder();
    }
    if (encoder == nil) {
        if (timed) {
            MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
            pass.sampleBufferAttachments[0].sampleBuffer = samples;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * ops.size();
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * ops.size() + 1;
            encoder = [commands computeCommandEncoderWithDescriptor:pass];
        } else {
            encoder = [commands computeCommandEncoder];
        }
        encoder_timed = timed;
    }
    return encoder;
}

void lumen_mps_stream_encoded(lumen_mps_completion done, void *context) {
    ops.push_back({done, context, encoder_timed ? (long)(2 * ops.size()) : -1});
    ++encoded_total;
    if (encoder_timed) {
        end_encoder();
    }
    if (ops.size() >= kOpsPerCommit) {
        commit();
    }
    os_unfair_lock_unlock(&lock);
}

static id<MTLEvent> barrier = nil;
static uint64_t barrier_value = 0;

extern "C" {
void lumen_mps_stream_fence(void) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    if (queue == nil) {
        return;
    }
    os_unfair_lock_lock(&lock);
    if (commands == nil) {
        commands = [queue commandBuffer];
    }
    if (encoder != nil) {
        end_encoder();
    }
    if (barrier == nil) {
        barrier = [queue.device newEvent];
    }
    ++barrier_value;
    [commands encodeSignalEvent:barrier value:barrier_value];
    [commands encodeWaitForEvent:barrier value:barrier_value];
    os_unfair_lock_unlock(&lock);
}

uint64_t lumen_mps_stream_flush(void) {
    os_unfair_lock_lock(&lock);
    commit();
    uint64_t total = encoded_total;
    os_unfair_lock_unlock(&lock);
    return total;
}

uint64_t lumen_mps_stream_encoded_total(void) {
    os_unfair_lock_lock(&lock);
    uint64_t total = encoded_total;
    os_unfair_lock_unlock(&lock);
    return total;
}

void *lumen_mps_stream_command_buffer(void) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    if (queue == nil) {
        return nullptr;
    }
    os_unfair_lock_lock(&lock);
    if (commands == nil) {
        commands = [queue commandBuffer];
    }
    if (encoder != nil) {
        end_encoder();
    }
    void *buffer = (__bridge void *)commands;
    os_unfair_lock_unlock(&lock);
    return buffer;
}

void lumen_mps_stream_mark(lumen_mps_completion done, void *context) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
    os_unfair_lock_lock(&lock);
    if (commands == nil) {
        commands = [queue commandBuffer];
    }
    ops.push_back({done, context, -1});
    ++encoded_total;
    if (ops.size() >= kOpsPerCommit) {
        commit();
    }
    os_unfair_lock_unlock(&lock);
}

double lumen_mps_stream_host_time(void) {
    static mach_timebase_info_data_t timebase;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        mach_timebase_info(&timebase);
    });
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
}
}
