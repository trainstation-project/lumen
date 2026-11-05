#import <Metal/Metal.h>

typedef void (*lumen_mps_completion)(void *context, double gpu_start, double gpu_end, int ok);

extern "C" void *lumen_mps_stream_queue(void);

id<MTLComputeCommandEncoder> lumen_mps_stream_encoder(bool timed);

void lumen_mps_stream_encoded(lumen_mps_completion done, void *context);
