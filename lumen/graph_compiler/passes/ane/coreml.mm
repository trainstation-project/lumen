
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <vector>

namespace {

struct Program {
    MLModel *model;
    NSURL *compiled;
    NSString *devices;
    NSArray<NSString *> *inputs;
    NSArray<NSString *> *outputs;
};

char *copy_string(NSString *s) {
    const char *utf8 = s.UTF8String ?: "";
    char *out = (char *)malloc(strlen(utf8) + 1);
    strcpy(out, utf8);
    return out;
}

void set_error(char **error, NSString *message) {
    if (error) {
        *error = copy_string(message);
    }
}

NSString *device_name(id<MLComputeDeviceProtocol> device) {
    if ([device isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
        return @"ane";
    }
    if ([device isKindOfClass:[MLGPUComputeDevice class]]) {
        return @"gpu";
    }
    return @"cpu";
}

NSString *devices_of(NSURL *compiled, MLModelConfiguration *config) {
    __block NSMutableString *lines = [NSMutableString string];
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled
                       configuration:config
                   completionHandler:^(MLComputePlan *plan, NSError *) {
                       MLModelStructureProgramFunction *main = plan.modelStructure.program.functions[@"main"];
                       for (MLModelStructureProgramOperation *op in main.block.operations) {
                           if ([op.operatorName isEqualToString:@"const"]) {
                               continue;
                           }
                           MLComputePlanDeviceUsage *usage = [plan computeDeviceUsageForMLProgramOperation:op];
                           NSString *device = usage ? device_name(usage.preferredComputeDevice) : @"none";
                           for (MLModelStructureProgramNamedValueType *output in op.outputs) {
                               [lines appendFormat:@"%@\t%@\t%@\n", output.name, op.operatorName, device];
                           }
                       }
                       dispatch_semaphore_signal(done);
                   }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    return lines;
}

MLMultiArrayDataType array_type(int32_t code) { return (MLMultiArrayDataType)code; }

size_t element_size(int32_t code) { return code == MLMultiArrayDataTypeFloat16 ? 2 : 4; }

} // namespace

extern "C" void *lumen_coreml_compile(const uint8_t *spec, size_t len, char **error) {
    @autoreleasepool {
        NSString *dir = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString];
        NSError *err = nil;
        if (![NSFileManager.defaultManager createDirectoryAtPath:dir
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:&err]) {
            set_error(error, err.localizedDescription);
            return nullptr;
        }
        NSURL *source = [NSURL fileURLWithPath:[dir stringByAppendingPathComponent:@"program.mlmodel"]];
        [[NSData dataWithBytesNoCopy:(void *)spec length:len freeWhenDone:NO] writeToURL:source atomically:NO];
        NSURL *compiled = [MLModel compileModelAtURL:source error:&err];
        [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
        if (!compiled) {
            set_error(error, err.description);
            return nullptr;
        }
        MLModelConfiguration *config = [MLModelConfiguration new];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:config error:&err];
        if (!model) {
            [NSFileManager.defaultManager removeItemAtURL:compiled error:nil];
            set_error(error, err.description);
            return nullptr;
        }
        auto *program = new Program();
        program->model = model;
        program->compiled = compiled;
        program->devices = devices_of(compiled, config);
        NSMutableArray<NSString *> *inputs = [NSMutableArray array], *outputs = [NSMutableArray array];
        for (NSUInteger k = 0; k < model.modelDescription.inputDescriptionsByName.count; ++k) {
            [inputs addObject:[NSString stringWithFormat:@"in%lu", (unsigned long)k]];
        }
        for (NSUInteger k = 0; k < model.modelDescription.outputDescriptionsByName.count; ++k) {
            [outputs addObject:[NSString stringWithFormat:@"out%lu", (unsigned long)k]];
        }
        program->inputs = inputs;
        program->outputs = outputs;
        return program;
    }
}

extern "C" char *lumen_coreml_devices(void *handle) { return copy_string(static_cast<Program *>(handle)->devices); }

static MLMultiArray *wrap(
    void *data, int32_t dtype, size_t rank, const int64_t *shape, const int64_t *strides, NSError **err) {
    NSMutableArray<NSNumber *> *dims = [NSMutableArray array], *steps = [NSMutableArray array];
    int64_t stride = 1;
    for (size_t d = rank; d-- > 0;) {
        [dims insertObject:@(shape[d]) atIndex:0];
        [steps insertObject:@(strides ? strides[d] : stride) atIndex:0];
        stride *= shape[d];
    }
    return [[MLMultiArray alloc] initWithDataPointer:data
                                               shape:dims
                                            dataType:array_type(dtype)
                                             strides:steps
                                         deallocator:nil
                                               error:err];
}

extern "C" int32_t lumen_coreml_predict(void *handle,
                                        size_t n_in,
                                        const uint8_t *const *in_data,
                                        const int32_t *in_dtypes,
                                        const size_t *in_ranks,
                                        const int64_t *in_shapes,
                                        const int64_t *in_strides,
                                        size_t n_out,
                                        uint8_t *const *out_data,
                                        const int32_t *out_dtypes,
                                        const size_t *out_ranks,
                                        const int64_t *out_shapes,
                                        char **error) {
    @autoreleasepool {
        auto *program = static_cast<Program *>(handle);
        NSError *err = nil;
        NSMutableDictionary<NSString *, id> *features = [NSMutableDictionary dictionary];
        for (size_t k = 0; k < n_in; ++k) {
            MLMultiArray *array = wrap((void *)in_data[k], in_dtypes[k], in_ranks[k], in_shapes, in_strides, &err);
            if (!array) {
                set_error(error, err.description);
                return 1;
            }
            features[program->inputs[k]] = array;
            in_shapes += in_ranks[k];
            in_strides += in_ranks[k];
        }
        MLPredictionOptions *options = [MLPredictionOptions new];
        NSMutableDictionary<NSString *, id> *backings = [NSMutableDictionary dictionary];
        const int64_t *shapes = out_shapes;
        for (size_t k = 0; k < n_out; ++k) {
            MLMultiArray *array = wrap(out_data[k], out_dtypes[k], out_ranks[k], shapes, nullptr, &err);
            if (!array) {
                set_error(error, err.description);
                return 1;
            }
            backings[program->outputs[k]] = array;
            shapes += out_ranks[k];
        }
        options.outputBackings = backings;
        MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features
                                                                                                  error:&err];
        id<MLFeatureProvider> result =
            provider ? [program->model predictionFromFeatures:provider options:options error:&err] : nil;
        if (!result) {
            set_error(error, err.description);
            return 1;
        }
        for (size_t k = 0; k < n_out; ++k) {
            MLMultiArray *array = [result featureValueForName:program->outputs[k]].multiArrayValue;
            if (!array || array.dataType != array_type(out_dtypes[k])) {
                set_error(error, [NSString stringWithFormat:@"output %zu is not of the program's type", k]);
                return 1;
            }
            const size_t size = element_size(out_dtypes[k]), rank = array.shape.count, count = array.count;
            std::vector<size_t> shape(rank), strides(rank);
            for (size_t d = 0; d < rank; ++d) {
                shape[d] = array.shape[d].unsignedLongValue;
                strides[d] = array.strides[d].unsignedLongValue;
            }
            uint8_t *out = out_data[k];
            const size_t row = rank && strides[rank - 1] == 1 ? shape[rank - 1] : 1;
            [array getBytesWithHandler:^(const void *bytes, NSInteger) {
                if (bytes == out) {
                    return;
                }
                const uint8_t *in = (const uint8_t *)bytes;
                for (size_t e = 0; e < count; e += row) {
                    size_t rest = e, offset = 0;
                    for (size_t d = rank; d-- > 0;) {
                        offset += (rest % shape[d]) * strides[d];
                        rest /= shape[d];
                    }
                    memcpy(out + e * size, in + offset * size, row * size);
                }
            }];
        }
        return 0;
    }
}

static id<MTLSharedEvent> event = nil;
static MTLSharedEventListener *listener = nil;

static id<MTLSharedEvent> event_for(id<MTLDevice> device) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        event = [device newSharedEvent];
        listener = [[MTLSharedEventListener alloc]
            initWithDispatchQueue:dispatch_queue_create("lumen.coreml.event", DISPATCH_QUEUE_SERIAL)];
    });
    return event;
}

extern "C" void lumen_coreml_encode_signal(void *command_buffer, uint64_t value) {
    id<MTLCommandBuffer> commands = (__bridge id<MTLCommandBuffer>)command_buffer;
    [commands encodeSignalEvent:event_for(commands.device) value:value];
}

extern "C" void lumen_coreml_encode_wait(void *command_buffer, uint64_t value) {
    id<MTLCommandBuffer> commands = (__bridge id<MTLCommandBuffer>)command_buffer;
    [commands encodeWaitForEvent:event_for(commands.device) value:value];
}

extern "C" void lumen_coreml_host_wait(uint64_t value) {
    if (event.signaledValue >= value) {
        return;
    }
    dispatch_semaphore_t reached = dispatch_semaphore_create(0);
    [event notifyListener:listener
                  atValue:value
                    block:^(id<MTLSharedEvent>, uint64_t) {
                        dispatch_semaphore_signal(reached);
                    }];
    dispatch_semaphore_wait(reached, DISPATCH_TIME_FOREVER);
}

extern "C" void lumen_coreml_host_signal(uint64_t value) { event.signaledValue = value; }

extern "C" void lumen_coreml_free(void *handle) {
    auto *program = static_cast<Program *>(handle);
    [NSFileManager.defaultManager removeItemAtURL:program->compiled error:nil];
    delete program;
}

extern "C" void lumen_coreml_free_string(char *s) { free(s); }
