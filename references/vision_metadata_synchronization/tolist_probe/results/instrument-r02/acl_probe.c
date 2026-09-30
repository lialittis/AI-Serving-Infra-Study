/* Process-local diagnostic interposition; never use these timings as benchmarks. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint64_t scope, op, begin_ns, end_ns, tid, src, dst, bytes, kind, stream;
    int64_t result;
} Row;
static Row rows[4096];
static unsigned used;
static __thread unsigned scope;
static void *frames[24];
static int frame_count;
static uint64_t now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
void probe_scope(unsigned value) { scope = value; }
unsigned probe_count(void) { return used; }
const Row *probe_rows(void) { return rows; }
void probe_stack(const char *path) {
    FILE *f = fopen(path, "w"); if (!f) abort();
    backtrace_symbols_fd(frames, frame_count, fileno(f)); fclose(f);
}
static void append(Row row) {
    if (used >= 4096) abort();
    rows[used++] = row;
}
int aclrtMemcpy(void *dst, size_t max, const void *src, size_t bytes, int kind) {
    typedef int (*Fn)(void *, size_t, const void *, size_t, int);
    static Fn next;
    if (!next) { next = (Fn)dlsym(RTLD_NEXT, "aclrtMemcpy"); if (!next) abort(); }
    if (!scope) return next(dst, max, src, bytes, kind);
    if (!frame_count && scope == 2) frame_count = backtrace(frames, 24);
    Row r = {.scope=scope, .op=2, .tid=syscall(SYS_gettid), .src=(uintptr_t)src,
             .dst=(uintptr_t)dst, .bytes=bytes, .kind=kind};
    r.begin_ns=now(); r.result=next(dst,max,src,bytes,kind); r.end_ns=now();
    append(r); return r.result;
}
int aclrtSynchronizeStream(void *stream) {
    typedef int (*Fn)(void *);
    static Fn next;
    if (!next) { next = (Fn)dlsym(RTLD_NEXT, "aclrtSynchronizeStream"); if (!next) abort(); }
    if (!scope) return next(stream);
    Row r = {.scope=scope, .op=1, .tid=syscall(SYS_gettid), .stream=(uintptr_t)stream};
    r.begin_ns=now(); r.result=next(stream); r.end_ns=now(); append(r); return r.result;
}
