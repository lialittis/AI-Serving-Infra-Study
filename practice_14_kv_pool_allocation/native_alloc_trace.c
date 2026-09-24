/* Observe this experiment's CANN allocation calls without changing arguments,
 * return values, allocation policy, or adding device synchronization.
 * Loaded only into the launched service with LD_AUDIT, never system-wide.
 * GNU audit also observes explicit dlsym(handle, name) bindings used by torch-npu. */
#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static const char *names[] = {"aclrtMalloc", "aclrtMallocAlign32", "aclrtMallocWithCfg",
    "aclrtReserveMemAddress", "aclrtMallocPhysical", "aclrtMapMem", "aclrtFree"};
static uintptr_t originals[7];

static void *resolve(const char *name) {
    for (size_t i = 0; i < 7; ++i)
        if (strcmp(name, names[i]) == 0) return (void *)__atomic_load_n(&originals[i], __ATOMIC_ACQUIRE);
    _exit(127);
}

static void record(const char *api, uint64_t begin, uint64_t end,
                   uintptr_t address, size_t size, int result, uintptr_t extra,
                   void *caller) {
    int saved_errno = errno;
    const char *directory = getenv("P14_NATIVE_DIR");
    if (directory) {
        char path[4096], line[2048];
        Dl_info info = {0};
        dladdr(caller, &info);
        int n = snprintf(path, sizeof(path), "%s/native-%d.jsonl", directory, getpid());
        if (n < 0 || (size_t)n >= sizeof(path)) _exit(126);
        int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        n = snprintf(line, sizeof(line),
            "{\"api\":\"%s\",\"pid\":%d,\"tid\":%ld,\"begin_ns\":%llu,"
            "\"end_ns\":%llu,\"address\":%llu,\"bytes\":%zu,\"result\":%d,"
            "\"extra\":%llu,\"caller_library\":\"%s\",\"caller_offset\":%llu}\n",
            api, getpid(), syscall(SYS_gettid), (unsigned long long)begin,
            (unsigned long long)end, (unsigned long long)address, size, result,
            (unsigned long long)extra, info.dli_fname ? info.dli_fname : "unknown",
            (unsigned long long)((uintptr_t)caller - (uintptr_t)info.dli_fbase));
        if (fd < 0 || n < 0 || (size_t)n >= sizeof(line) || write(fd, line, n) != n) _exit(126);
        close(fd);
    }
    errno = saved_errno;
}

#define MALLOC_WRAPPER(NAME) \
aclError NAME(void **ptr, size_t size, aclrtMemMallocPolicy policy) { \
    typedef aclError (*fn_t)(void **, size_t, aclrtMemMallocPolicy); \
    fn_t fn = (fn_t)resolve(#NAME); uint64_t begin = now_ns(); \
    aclError result = fn(ptr, size, policy); uint64_t end = now_ns(); \
    record(#NAME, begin, end, result == ACL_SUCCESS ? (uintptr_t)*ptr : 0, size, \
           result, policy, __builtin_return_address(0)); return result; \
}
MALLOC_WRAPPER(aclrtMalloc)
MALLOC_WRAPPER(aclrtMallocAlign32)

aclError aclrtMallocWithCfg(void **ptr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig *cfg) {
    typedef aclError (*fn_t)(void **, size_t, aclrtMemMallocPolicy, aclrtMallocConfig *);
    fn_t fn = (fn_t)resolve("aclrtMallocWithCfg"); uint64_t begin = now_ns();
    aclError result = fn(ptr, size, policy, cfg); uint64_t end = now_ns();
    record("aclrtMallocWithCfg", begin, end, result == ACL_SUCCESS ? (uintptr_t)*ptr : 0,
           size, result, policy, __builtin_return_address(0)); return result;
}

aclError aclrtReserveMemAddress(void **ptr, size_t size, size_t alignment, void *expected, uint64_t flags) {
    typedef aclError (*fn_t)(void **, size_t, size_t, void *, uint64_t);
    fn_t fn = (fn_t)resolve("aclrtReserveMemAddress"); uint64_t begin = now_ns();
    aclError result = fn(ptr, size, alignment, expected, flags); uint64_t end = now_ns();
    record("aclrtReserveMemAddress", begin, end, result == ACL_SUCCESS ? (uintptr_t)*ptr : 0,
           size, result, alignment, __builtin_return_address(0)); return result;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle *handle, size_t size, const aclrtPhysicalMemProp *prop, uint64_t flags) {
    typedef aclError (*fn_t)(aclrtDrvMemHandle *, size_t, const aclrtPhysicalMemProp *, uint64_t);
    fn_t fn = (fn_t)resolve("aclrtMallocPhysical"); uint64_t begin = now_ns();
    aclError result = fn(handle, size, prop, flags); uint64_t end = now_ns();
    record("aclrtMallocPhysical", begin, end, result == ACL_SUCCESS ? (uintptr_t)*handle : 0,
           size, result, flags, __builtin_return_address(0)); return result;
}

aclError aclrtMapMem(void *ptr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags) {
    typedef aclError (*fn_t)(void *, size_t, size_t, aclrtDrvMemHandle, uint64_t);
    fn_t fn = (fn_t)resolve("aclrtMapMem"); uint64_t begin = now_ns();
    aclError result = fn(ptr, size, offset, handle, flags); uint64_t end = now_ns();
    record("aclrtMapMem", begin, end, (uintptr_t)ptr, size, result, (uintptr_t)handle,
           __builtin_return_address(0)); return result;
}

aclError aclrtFree(void *ptr) {
    typedef aclError (*fn_t)(void *);
    fn_t fn = (fn_t)resolve("aclrtFree"); uint64_t begin = now_ns();
    aclError result = fn(ptr); uint64_t end = now_ns();
    record("aclrtFree", begin, end, (uintptr_t)ptr, 0, result, 0,
           __builtin_return_address(0)); return result;
}

/* Preserve the loader's exact resolved function. The wrappers call it once.
 * Refuse ambiguous multiple implementations rather than silently redirecting. */
unsigned int la_version(unsigned int version) { (void)version; return LAV_CURRENT; }
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
    (void)lmid; *cookie = (uintptr_t)map; return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}
uintptr_t la_symbind64(Elf64_Sym *symbol, unsigned int index, uintptr_t *ref,
                     uintptr_t *def, unsigned int *flags, const char *name) {
    (void)index; (void)def; (void)flags;
    const char *requester = ((struct link_map *)*ref)->l_name;
    if (!strstr(requester, "torch_npu")) return symbol->st_value;
    void *wrappers[] = {(void *)aclrtMalloc, (void *)aclrtMallocAlign32,
        (void *)aclrtMallocWithCfg, (void *)aclrtReserveMemAddress,
        (void *)aclrtMallocPhysical, (void *)aclrtMapMem, (void *)aclrtFree};
    for (size_t i = 0; i < 7; ++i) {
        if (strcmp(name, names[i]) != 0) continue;
        uintptr_t previous = __atomic_load_n(&originals[i], __ATOMIC_ACQUIRE);
        if (previous && previous != symbol->st_value) { fprintf(stderr, "P14 ambiguous binding: %s from %s\n", name, requester); _exit(125); }
        __atomic_store_n(&originals[i], symbol->st_value, __ATOMIC_RELEASE);
        return (uintptr_t)wrappers[i];
    }
    return symbol->st_value;
}
