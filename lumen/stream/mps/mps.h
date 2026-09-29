// lumen's MPS stream, for ops' Objective-C++ shims (stream/mps/mps.mm).

#import <Metal/Metal.h>

// Called once the GPU has run an op: its GPU start and end in host-clock
// seconds, and ok = 0 if its command buffer failed.
typedef void (*lumen_mps_completion)(void *context, double gpu_start, double gpu_end, int ok);

extern "C" void *lumen_mps_stream_queue(void);

// The stream's open compute encoder, for one op to encode into; nil without
// a Metal device. Locks the stream until lumen_mps_stream_encoded, which
// must follow on the same thread. timed gives the op an encoder of its own
// whose GPU start and end are sampled, for the profiler.
id<MTLComputeCommandEncoder> lumen_mps_stream_encoder(bool timed);

// The op is encoded: done(context, ...) is called once the GPU has run it.
void lumen_mps_stream_encoded(lumen_mps_completion done, void *context);
