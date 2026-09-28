// lumen's MPS stream (PyTorch: MPSStream): the one command queue MPS ops
// submit GPU work to. Ops commit their command buffers without waiting; the
// host waits only in stream::mps::synchronize (stream/mps.rs), before it
// touches MPS memory. build.rs compiles this file on macOS.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>

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
