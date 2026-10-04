#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <os/lock.h>

#include <vector>

#include "mps.h"

extern "C" {
// The stream's command queue (unretained; it lives for the process), or
// null without a Metal device.
void *lumen_mps_stream_queue(void) {
    static id<MTLCommandQueue> queue = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        queue = [MTLCreateSystemDefaultDevice() newCommandQueue];
    });
    return (__bridge void *)queue;
}
}

// Committed at this many ops, so the GPU starts on them and their tensors
// are released as they finish.
static constexpr size_t kOpsPerCommit = 32;

struct Op {
    lumen_mps_completion done;
    void *context;
    long sample; // index of its start sample, or -1 if not timed
};

// The open command buffer, guarded by lock.
static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
static id<MTLCommandBuffer> commands = nil;
static id<MTLComputeCommandEncoder> encoder = nil;
static bool encoder_timed = false;
static id<MTLCounterSampleBuffer> samples = nil; // timed ops' start/end
static std::vector<Op> ops;
static uint64_t encoded_total = 0; // ops ever encoded

static void end_encoder(void) {
    [encoder endEncoding];
    encoder = nil;
}

// A sample buffer for kOpsPerCommit timed ops' GPU timestamps, or nil.
static id<MTLCounterSampleBuffer> new_samples(id<MTLDevice> device) {
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

// Commit the open command buffer; its handler calls every op's done.
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
    // One reading of the CPU (mach_absolute_time ns) and GPU clocks, to map
    // the samples to host time.
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
            // Its own encoder, sampled at its start and end.
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
        end_encoder(); // so its end is sampled now
    }
    if (ops.size() >= kOpsPerCommit) {
        commit();
    }
    os_unfair_lock_unlock(&lock);
}

extern "C" {
// Commit the ops encoded so far, returning how many ops have ever been
// encoded (stream::mps::synchronize waits until that many have finished).
uint64_t lumen_mps_stream_flush(void) {
    os_unfair_lock_lock(&lock);
    commit();
    uint64_t total = encoded_total;
    os_unfair_lock_unlock(&lock);
    return total;
}

// The open command buffer (made if none is), its compute encoder ended so
// the caller may encode its own work into it, after the ops encoded so far
// and before those encoded after: valid until the stream commits it (the
// next flush, or op that fills it). Null without a Metal device.
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

// Count work encoded into the open command buffer outside lumen's ops (a
// custom op's, through lumen_mps_stream_command_buffer) as an op: done is
// called when the buffer completes, which synchronize then waits for.
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

// The host clock Metal's GPUStartTime/GPUEndTime use (mach_absolute_time, as
// seconds), so GPU times can be mapped onto the profiler's clock.
double lumen_mps_stream_host_time(void) {
    static mach_timebase_info_data_t timebase;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        mach_timebase_info(&timebase);
    });
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
}
}
