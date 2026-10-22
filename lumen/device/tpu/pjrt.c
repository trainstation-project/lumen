
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"
#include "profiler_c_api.h"

static const PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *const *devices;
static size_t device_count;
static char error[1024];
static pthread_once_t once = PTHREAD_ONCE_INIT;
static atomic_int opened;

void lumen_tpu_free(void *buffer);

#define HAS(field) (offsetof(PJRT_Api, field) + sizeof(api->field) <= api->struct_size && api->field)

static int failed(PJRT_Error *e, const char *what) {
    if (!e) {
        return 0;
    }
    PJRT_Error_Message_Args message = {.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = e};
    api->PJRT_Error_Message(&message);
    snprintf(error, sizeof error, "%s: %.*s", what, (int)message.message_size, message.message);
    PJRT_Error_Destroy_Args destroy = {.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE, .error = e};
    api->PJRT_Error_Destroy(&destroy);
    return 1;
}

static int awaited(PJRT_Event *event, const char *what) {
    if (!event) {
        return 0;
    }
    PJRT_Event_Await_Args await = {.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = event};
    int bad = failed(api->PJRT_Event_Await(&await), what);
    PJRT_Event_Destroy_Args destroy = {.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = event};
    api->PJRT_Event_Destroy(&destroy);
    return bad;
}

static void initialize(void) {
    atomic_store(&opened, 1);
    const char *path = getenv("TPU_LIBRARY_PATH");
    if (!path || !*path) {
        path = "libtpu.so";
    }
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        snprintf(error, sizeof error, "loading %s (set TPU_LIBRARY_PATH to libtpu.so): %s", path, dlerror());
        return;
    }
    const PJRT_Api *(*get)(void) = (const PJRT_Api *(*)(void))dlsym(library, "GetPjrtApi");
    if (!get || !(api = get())) {
        snprintf(error, sizeof error, "%s has no PJRT API (GetPjrtApi)", path);
        api = NULL;
        return;
    }
    if (!(HAS(PJRT_Error_Message) && HAS(PJRT_Error_Destroy) && HAS(PJRT_Event_Await) && HAS(PJRT_Event_Destroy) &&
          HAS(PJRT_Client_Create) && HAS(PJRT_Client_AddressableDevices) &&
          HAS(PJRT_Client_CreateUninitializedBuffer) && HAS(PJRT_Buffer_UnsafePointer) &&
          HAS(PJRT_Buffer_CopyRawToHost) && HAS(PJRT_Buffer_Destroy) && HAS(PJRT_Client_Compile) &&
          HAS(PJRT_Client_BufferFromHostBuffer) && HAS(PJRT_LoadedExecutable_Execute) && HAS(PJRT_Buffer_ReadyEvent) &&
          HAS(PJRT_Device_GetDescription) && HAS(PJRT_DeviceDescription_Id))) {
        snprintf(error,
                 sizeof error,
                 "%s's PJRT API (%d.%d) lacks a function lumen needs",
                 path,
                 api->pjrt_api_version.major_version,
                 api->pjrt_api_version.minor_version);
        api = NULL;
        return;
    }
    if (HAS(PJRT_Plugin_Initialize)) {
        PJRT_Plugin_Initialize_Args args = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
        if (failed(api->PJRT_Plugin_Initialize(&args), "initializing libtpu")) {
            return;
        }
    }
    PJRT_Client_Create_Args create = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
    if (failed(api->PJRT_Client_Create(&create), "creating the TPU client")) {
        return;
    }
    PJRT_Client_AddressableDevices_Args addressable = {.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
                                                       .client = create.client};
    if (failed(api->PJRT_Client_AddressableDevices(&addressable), "listing the TPU devices")) {
        return;
    }
    client = create.client;
    devices = addressable.addressable_devices;
    device_count = addressable.num_addressable_devices;
}

int lumen_tpu_opened(void) { return atomic_load(&opened); }

int lumen_tpu_device_count(void) {
    pthread_once(&once, initialize);
    return client ? (int)device_count : 0;
}

const char *lumen_tpu_error(void) {
    pthread_once(&once, initialize);
    return error;
}

#define ROW 512

void *lumen_tpu_alloc(int index, size_t nbytes, uintptr_t *address) {
    if (lumen_tpu_device_count() <= index || index < 0) {
        snprintf(error, sizeof error, "no TPU device %d", index);
        return NULL;
    }
    int64_t dims[3] = {(int64_t)(nbytes / ROW), 1, ROW / 4};
    PJRT_Client_CreateUninitializedBuffer_Args create = {
        .struct_size = PJRT_Client_CreateUninitializedBuffer_Args_STRUCT_SIZE,
        .client = client,
        .shape_dims = dims,
        .shape_num_dims = 3,
        .shape_element_type = PJRT_Buffer_Type_U32,
        .device = devices[index],
    };
    if (failed(api->PJRT_Client_CreateUninitializedBuffer(&create), "reserving TPU memory")) {
        return NULL;
    }
    PJRT_Buffer_UnsafePointer_Args pointer = {.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE,
                                              .buffer = create.buffer};
    if (failed(api->PJRT_Buffer_UnsafePointer(&pointer), "addressing TPU memory")) {
        lumen_tpu_free(create.buffer);
        return NULL;
    }
    *address = pointer.buffer_pointer;
    return create.buffer;
}

void lumen_tpu_free(void *buffer) {
    PJRT_Buffer_Destroy_Args destroy = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE,
                                        .buffer = (PJRT_Buffer *)buffer};
    failed(api->PJRT_Buffer_Destroy(&destroy), "freeing TPU memory");
}

#define DMA 32

int lumen_tpu_read(void *buffer, size_t size, size_t offset, void *dst, size_t nbytes) {
    if (nbytes == 0) {
        return 0;
    }
    size_t start = offset - offset % DMA, end = offset + nbytes;
    end = end + (DMA - end % DMA) % DMA;
    if (end > size) {
        end = size;
    }
    void *staging = aligned_alloc(DMA, end - start + DMA);
    if (!staging) {
        snprintf(error, sizeof error, "out of host memory staging a TPU read");
        return 1;
    }
    PJRT_Buffer_CopyRawToHost_Args copy = {.struct_size = PJRT_Buffer_CopyRawToHost_Args_STRUCT_SIZE,
                                           .buffer = (PJRT_Buffer *)buffer,
                                           .dst = staging,
                                           .offset = (int64_t)start,
                                           .transfer_size = (int64_t)(end - start)};
    int bad = failed(api->PJRT_Buffer_CopyRawToHost(&copy), "reading TPU memory") ||
              awaited(copy.event, "reading TPU memory");
    if (!bad) {
        memcpy(dst, (char *)staging + (offset - start), nbytes);
    }
    free(staging);
    return bad;
}

void *lumen_tpu_compile(const char *code, size_t size, int index) {
    if (lumen_tpu_device_count() <= index || index < 0) {
        snprintf(error, sizeof error, "no TPU device %d", index);
        return NULL;
    }
    PJRT_Device_GetDescription_Args describe = {.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
                                                .device = devices[index]};
    if (failed(api->PJRT_Device_GetDescription(&describe), "describing a TPU device")) {
        return NULL;
    }
    PJRT_DeviceDescription_Id_Args id = {.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE,
                                         .device_description = describe.device_description};
    if (failed(api->PJRT_DeviceDescription_Id(&id), "describing a TPU device")) {
        return NULL;
    }
    PJRT_Program program = {.struct_size = PJRT_Program_STRUCT_SIZE,
                            .code = (char *)code,
                            .code_size = size,
                            .format = "mlir",
                            .format_size = 4};
    char id_bytes[10];
    size_t id_size = 0;
    for (uint64_t v = (uint64_t)id.id; id_size == 0 || v; v >>= 7) {
        id_bytes[id_size++] = (char)((v & 0x7f) | (v >> 7 ? 0x80 : 0));
    }
    char options[32];
    size_t n = 0;
    options[n++] = 0x1a;
    options[n++] = (char)(13 + id_size);
    options[n++] = 0x20, options[n++] = 0x01, options[n++] = 0x28, options[n++] = 0x01;
    options[n++] = 0x4a;
    options[n++] = (char)(7 + id_size);
    options[n++] = 0x08, options[n++] = 0x01, options[n++] = 0x10, options[n++] = 0x01;
    options[n++] = 0x1a;
    options[n++] = (char)(1 + id_size);
    options[n++] = 0x08;
    memcpy(options + n, id_bytes, id_size);
    n += id_size;
    PJRT_Client_Compile_Args compile = {.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE,
                                        .client = client,
                                        .program = &program,
                                        .compile_options = options,
                                        .compile_options_size = n};
    if (failed(api->PJRT_Client_Compile(&compile), "compiling a TPU program")) {
        return NULL;
    }
    return compile.executable;
}

static PJRT_Buffer *upload(int index, const void *data, PJRT_Buffer_Type type, const int64_t *dims, size_t num_dims) {
    PJRT_Client_BufferFromHostBuffer_Args create = {
        .struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
        .client = client,
        .data = data,
        .type = type,
        .dims = dims,
        .num_dims = num_dims,
        .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
        .device = devices[index],
    };
    if (failed(api->PJRT_Client_BufferFromHostBuffer(&create), "copying to the TPU") ||
        awaited(create.done_with_host_buffer, "copying to the TPU")) {
        return NULL;
    }
    return create.buffer;
}

#define SCALARS 1024
#define CACHED 256
static struct {
    PJRT_Buffer *buffer;
    int index;
    size_t count;
    int32_t values[SCALARS];
} scalar_buffers[CACHED];
static size_t next_scalar_buffer;
static pthread_mutex_t scalars_lock = PTHREAD_MUTEX_INITIALIZER;

static PJRT_Buffer *scalars_buffer(int index, const int32_t *scalars, size_t count, int *owned) {
    int64_t length = (int64_t)count;
    *owned = count > SCALARS;
    if (*owned) {
        return upload(index, scalars, PJRT_Buffer_Type_S32, &length, 1);
    }
    pthread_mutex_lock(&scalars_lock);
    PJRT_Buffer *buffer = NULL;
    for (size_t i = 0; i < CACHED && !buffer; i++) {
        if (scalar_buffers[i].buffer && scalar_buffers[i].index == index && scalar_buffers[i].count == count &&
            memcmp(scalar_buffers[i].values, scalars, count * sizeof *scalars) == 0) {
            buffer = scalar_buffers[i].buffer;
        }
    }
    if (!buffer && (buffer = upload(index, scalars, PJRT_Buffer_Type_S32, &length, 1))) {
        size_t i = next_scalar_buffer++ % CACHED;
        if (scalar_buffers[i].buffer) {
            lumen_tpu_free(scalar_buffers[i].buffer);
        }
        scalar_buffers[i].buffer = buffer;
        scalar_buffers[i].index = index;
        scalar_buffers[i].count = count;
        memcpy(scalar_buffers[i].values, scalars, count * sizeof *scalars);
    }
    pthread_mutex_unlock(&scalars_lock);
    return buffer;
}

int lumen_tpu_run(void *executable,
                  int index,
                  const int32_t *scalars,
                  size_t count,
                  const void *host,
                  size_t nbytes,
                  void **region) {
    if (lumen_tpu_device_count() <= index || index < 0) {
        snprintf(error, sizeof error, "no TPU device %d", index);
        return 1;
    }
    PJRT_Buffer *args[3];
    size_t num_args = 0;
    int owned;
    if (!(args[num_args++] = scalars_buffer(index, scalars, count, &owned))) {
        return 1;
    }
    int64_t rows[3] = {(int64_t)(nbytes / ROW), 1, ROW / 4};
    if (nbytes && !(args[num_args++] = upload(index, host, PJRT_Buffer_Type_U32, rows, 3))) {
        if (owned) {
            lumen_tpu_free(args[0]);
        }
        return 1;
    }
    args[num_args++] = (PJRT_Buffer *)*region;
    PJRT_Buffer *result = NULL;
    PJRT_Buffer **results = &result;
    PJRT_Buffer *const *lists = args;
    PJRT_ExecuteOptions options = {.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE};
    PJRT_LoadedExecutable_Execute_Args execute = {.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
                                                  .executable = (PJRT_LoadedExecutable *)executable,
                                                  .options = &options,
                                                  .argument_lists = &lists,
                                                  .num_devices = 1,
                                                  .num_args = num_args,
                                                  .output_lists = &results};
    int bad = failed(api->PJRT_LoadedExecutable_Execute(&execute), "running a TPU program");
    for (size_t i = owned ? 0 : 1; i + 1 < num_args; i++) {
        lumen_tpu_free(args[i]);
    }
    if (bad) {
        return 1;
    }
    lumen_tpu_free(*region);
    *region = result;
    return 0;
}

int lumen_tpu_wait(void *buffer) {
    PJRT_Buffer_ReadyEvent_Args ready = {.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE,
                                         .buffer = (PJRT_Buffer *)buffer};
    return failed(api->PJRT_Buffer_ReadyEvent(&ready), "waiting for a TPU program") ||
           awaited(ready.event, "waiting for a TPU program");
}

int lumen_tpu_memory(int index, int64_t *in_use, int64_t *limit) {
    if (lumen_tpu_device_count() <= index || index < 0 || !HAS(PJRT_Device_MemoryStats)) {
        snprintf(error, sizeof error, "no memory statistics for TPU device %d", index);
        return 1;
    }
    PJRT_Device_MemoryStats_Args stats = {.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE,
                                          .device = devices[index]};
    if (failed(api->PJRT_Device_MemoryStats(&stats), "reading TPU memory statistics")) {
        return 1;
    }
    *in_use = stats.bytes_in_use;
    *limit = stats.bytes_limit;
    return 0;
}

typedef struct {
    PJRT_Extension_Base base;
    PLUGIN_Profiler_Api *profiler_api;
    int64_t traceme_context_id;
} PJRT_Profiler_Extension;

static PLUGIN_Profiler *profiler;

static int profiler_failed(const PLUGIN_Profiler_Api *p, PLUGIN_Profiler_Error *e, const char *what) {
    if (!e) {
        return 0;
    }
    PLUGIN_Profiler_Error_Message_Args message = {.struct_size = PLUGIN_Profiler_Error_Message_Args_STRUCT_SIZE,
                                                  .error = e};
    p->error_message(&message);
    snprintf(error, sizeof error, "%s: %.*s", what, (int)message.message_size, message.message);
    PLUGIN_Profiler_Error_Destroy_Args destroy = {.struct_size = PLUGIN_Profiler_Error_Destroy_Args_STRUCT_SIZE,
                                                  .error = e};
    p->error_destroy(&destroy);
    return 1;
}

static const PLUGIN_Profiler_Api *profiler_api(void) {
    if (!lumen_tpu_device_count()) {
        return NULL;
    }
    for (const PJRT_Extension_Base *e = api->extension_start; e; e = e->next) {
        if (e->type == PJRT_Extension_Type_Profiler) {
            return ((const PJRT_Profiler_Extension *)e)->profiler_api;
        }
    }
    snprintf(error, sizeof error, "libtpu has no profiler (PJRT's profiler extension)");
    return NULL;
}

int lumen_tpu_profiler_start(void) {
    const PLUGIN_Profiler_Api *p = profiler_api();
    if (!p) {
        return 1;
    }
    PLUGIN_Profiler_Create_Args create = {.struct_size = PLUGIN_Profiler_Create_Args_STRUCT_SIZE, .options = ""};
    if (profiler_failed(p, p->create(&create), "creating the TPU profiler")) {
        return 1;
    }
    profiler = create.profiler;
    PLUGIN_Profiler_Start_Args start = {.struct_size = PLUGIN_Profiler_Start_Args_STRUCT_SIZE, .profiler = profiler};
    return profiler_failed(p, p->start(&start), "starting the TPU profiler");
}

int lumen_tpu_profiler_stop(const uint8_t **data, size_t *size) {
    const PLUGIN_Profiler_Api *p = profiler_api();
    PLUGIN_Profiler_Stop_Args stop = {.struct_size = PLUGIN_Profiler_Stop_Args_STRUCT_SIZE, .profiler = profiler};
    if (profiler_failed(p, p->stop(&stop), "stopping the TPU profiler")) {
        return 1;
    }
    PLUGIN_Profiler_CollectData_Args collect = {.struct_size = PLUGIN_Profiler_CollectData_Args_STRUCT_SIZE,
                                                .profiler = profiler};
    if (profiler_failed(p, p->collect_data(&collect), "collecting the TPU profile")) {
        return 1;
    }
    *data = collect.buffer;
    *size = collect.buffer_size_in_bytes ? collect.buffer_size_in_bytes - 1 : 0;
    return 0;
}

void lumen_tpu_profiler_free(void) {
    const PLUGIN_Profiler_Api *p = profiler_api();
    PLUGIN_Profiler_Destroy_Args destroy = {.struct_size = PLUGIN_Profiler_Destroy_Args_STRUCT_SIZE,
                                            .profiler = profiler};
    profiler_failed(p, p->destroy(&destroy), "freeing the TPU profiler");
    profiler = NULL;
}
