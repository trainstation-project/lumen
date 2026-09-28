// lumen's MPS stream (PyTorch: MPSStream): the one command queue MPS ops
// submit GPU work to. Ops commit their command buffers without waiting; the
// host waits only in stream::mps::synchronize (stream/mps.rs), before it
// touches MPS memory. build.rs compiles this file on macOS.
//
// Work runs in submission order, like a CUDA stream: Metal orders command
// buffers only through the MTLBuffers they share, and ops bind their own
// no-copy views of tensor memory, so without an order two fills of the same
// tensor could run at once. Each command buffer waits on a shared event the
// one before it signals.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <os/lock.h>

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

static os_unfair_lock order_lock = OS_UNFAIR_LOCK_INIT;
static id<MTLSharedEvent> order_event = nil;
static uint64_t order_value = 0; // signaled by the last committed buffer

// A command buffer on the stream that starts after all work committed
// before it, or nil without a Metal device. Encode into it, then pass it to
// lumen_mps_stream_commit on the same thread; the stream is locked between
// the two, so buffers are committed in the order they wait.
id<MTLCommandBuffer> lumen_mps_stream_begin(void) {
  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lumen_mps_stream_queue();
  if (queue == nil) {
    return nil;
  }
  os_unfair_lock_lock(&order_lock);
  if (order_event == nil) {
    order_event = [queue.device newSharedEvent];
  }
  id<MTLCommandBuffer> commands = [queue commandBuffer];
  [commands encodeWaitForEvent:order_event value:order_value];
  return commands;
}

// Commit a buffer from lumen_mps_stream_begin, without waiting.
void lumen_mps_stream_commit(id<MTLCommandBuffer> commands) {
  [commands encodeSignalEvent:order_event value:++order_value];
  [commands commit];
  os_unfair_lock_unlock(&order_lock);
}

extern "C" {

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
