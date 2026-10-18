// libtpu's PJRT C API (`pjrt_c_api.h`, vendored from XLA), loaded at run
// time: the process's one PJRT client, its TPU devices, and the device
// memory the TPU allocator reserves and reads (lumen/allocator/tpu.rs).
//
// libtpu is found at `TPU_LIBRARY_PATH` (as JAX and torch_xla find it), else
// by the loader's search for `libtpu.so`. Its PJRT API may be older than this
// header: each function is used only if the plugin's table has it, and each
// argument struct carries this header's size, which a plugin accepts if it is
// at least its own.

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"

static const PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *const *devices;
static size_t device_count;
static char error[1024];
static pthread_once_t once = PTHREAD_ONCE_INIT;

void lumen_tpu_free(void *buffer);

// Whether the plugin's function table has `field` (an older plugin's is
// shorter).
#define HAS(field) (offsetof(PJRT_Api, field) + sizeof(api->field) <= api->struct_size && api->field)

// Records `e`'s message (prefixed by `what`) and frees it: whether it is one.
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

// Waits for `event` (none: done) and frees it: whether it failed.
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

// Loads libtpu and creates the client, once; on failure `error` says why and
// there are no devices.
static void initialize(void) {
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
          HAS(PJRT_Buffer_CopyRawToHost) && HAS(PJRT_Buffer_Destroy))) {
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

int lumen_tpu_device_count(void) {
    pthread_once(&once, initialize);
    return client ? (int)device_count : 0;
}

const char *lumen_tpu_error(void) {
    pthread_once(&once, initialize);
    return error;
}

// `nbytes` (a multiple of 4) of device `index`'s memory: a 1-D u32 buffer,
// which a TPU stores byte for byte as the host does (a u8 buffer's bytes are
// packed in tiles). Its handle, and the runtime's address of its memory in
// `*address`; none (and `error`) if it cannot be reserved.
void *lumen_tpu_alloc(int index, size_t nbytes, uintptr_t *address) {
    if (lumen_tpu_device_count() <= index || index < 0) {
        snprintf(error, sizeof error, "no TPU device %d", index);
        return NULL;
    }
    int64_t words = (int64_t)(nbytes / 4);
    PJRT_Client_CreateUninitializedBuffer_Args create = {
        .struct_size = PJRT_Client_CreateUninitializedBuffer_Args_STRUCT_SIZE,
        .client = client,
        .shape_dims = &words,
        .shape_num_dims = 1,
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

// The DMA alignment of a raw copy's host memory and offset.
#define DMA 32

// Copies `nbytes` at `offset` of `buffer` (an `lumen_tpu_alloc`'s, of
// `size` bytes) to `dst`, waiting for it: through host memory aligned for the
// TPU's DMA, the copy's range widened to aligned bounds inside the buffer.
// Whether it failed (`error` says why).
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

// Device `index`'s memory in use and its limit, in bytes: whether it failed.
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
