/* Process-local LD_AUDIT observer. Original calls run once, unchanged.
 * Read-only stream ID queries are performed separately by the Python observer.
 * No device waits, stream creation, resource limits or memory reads added. */
#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static uintptr_t originals[128];
static uint64_t now_ns(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec;}
static __thread int busy;
static void record(const char *api,uint64_t begin,uint64_t finish,uintptr_t stream,uintptr_t object,uint64_t extra,int rc,void *caller) {
    if(busy) return;
    busy=1;int saved_errno=errno;
    const char *dir=getenv("P22_NATIVE_DIR");
    if(dir) {
        int sid=-1;
        if(rc==0 && !strcmp(api,"aclrtStreamGetId"))sid=(int)object;
        char path[4096],line[2048];Dl_info info={0};dladdr(caller,&info);
        int n=snprintf(path,sizeof(path),"%s/native-%d-%ld.jsonl",dir,getpid(),syscall(SYS_gettid));
        if(n<0||(size_t)n>=sizeof(path))_exit(126);
        int fd=open(path,O_WRONLY|O_CREAT|O_APPEND|O_CLOEXEC,0644);
        n=snprintf(line,sizeof(line),"{\"api\":\"%s\",\"pid\":%d,\"tid\":%ld,\"begin_ns\":%llu,\"end_ns\":%llu,\"stream\":\"%llu\",\"object\":\"%llu\",\"extra\":%llu,\"result\":%d,\"stream_id\":%d,\"caller_address\":\"%llu\",\"caller_library\":\"%s\",\"caller_offset\":\"%llu\"}\n",api,getpid(),syscall(SYS_gettid),(unsigned long long)begin,(unsigned long long)finish,(unsigned long long)stream,(unsigned long long)object,(unsigned long long)extra,rc,sid,(unsigned long long)(uintptr_t)caller,info.dli_fname?info.dli_fname:"unknown",(unsigned long long)((uintptr_t)caller-(uintptr_t)info.dli_fbase));
        if(fd<0||n<0||(size_t)n>=sizeof(line)||write(fd,line,n)!=n)_exit(126);
        close(fd);
    }
    errno=saved_errno;busy=0;
}
static aclError p22_aclrtCreateStream_0(aclrtStream *s) {
 __typeof__(&aclrtCreateStream) fn=(__typeof__(&aclrtCreateStream))originals[0]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCreateStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStream_1(aclrtStream *s) {
 __typeof__(&aclrtCreateStream) fn=(__typeof__(&aclrtCreateStream))originals[1]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCreateStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStream_2(aclrtStream *s) {
 __typeof__(&aclrtCreateStream) fn=(__typeof__(&aclrtCreateStream))originals[2]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCreateStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStream_3(aclrtStream *s) {
 __typeof__(&aclrtCreateStream) fn=(__typeof__(&aclrtCreateStream))originals[3]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCreateStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamWithConfig_0(aclrtStream *s, uint32_t priority, uint32_t flag) {
 __typeof__(&aclrtCreateStreamWithConfig) fn=(__typeof__(&aclrtCreateStreamWithConfig))originals[4]; uint64_t begin=now_ns();
 aclError rc=fn(s,priority,flag); uint64_t finish=now_ns();
 record("aclrtCreateStreamWithConfig",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(priority),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamWithConfig_1(aclrtStream *s, uint32_t priority, uint32_t flag) {
 __typeof__(&aclrtCreateStreamWithConfig) fn=(__typeof__(&aclrtCreateStreamWithConfig))originals[5]; uint64_t begin=now_ns();
 aclError rc=fn(s,priority,flag); uint64_t finish=now_ns();
 record("aclrtCreateStreamWithConfig",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(priority),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamWithConfig_2(aclrtStream *s, uint32_t priority, uint32_t flag) {
 __typeof__(&aclrtCreateStreamWithConfig) fn=(__typeof__(&aclrtCreateStreamWithConfig))originals[6]; uint64_t begin=now_ns();
 aclError rc=fn(s,priority,flag); uint64_t finish=now_ns();
 record("aclrtCreateStreamWithConfig",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(priority),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamWithConfig_3(aclrtStream *s, uint32_t priority, uint32_t flag) {
 __typeof__(&aclrtCreateStreamWithConfig) fn=(__typeof__(&aclrtCreateStreamWithConfig))originals[7]; uint64_t begin=now_ns();
 aclError rc=fn(s,priority,flag); uint64_t finish=now_ns();
 record("aclrtCreateStreamWithConfig",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(priority),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamV2_0(aclrtStream *s, const aclrtStreamConfigHandle *cfg) {
 __typeof__(&aclrtCreateStreamV2) fn=(__typeof__(&aclrtCreateStreamV2))originals[8]; uint64_t begin=now_ns();
 aclError rc=fn(s,cfg); uint64_t finish=now_ns();
 record("aclrtCreateStreamV2",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(cfg),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamV2_1(aclrtStream *s, const aclrtStreamConfigHandle *cfg) {
 __typeof__(&aclrtCreateStreamV2) fn=(__typeof__(&aclrtCreateStreamV2))originals[9]; uint64_t begin=now_ns();
 aclError rc=fn(s,cfg); uint64_t finish=now_ns();
 record("aclrtCreateStreamV2",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(cfg),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamV2_2(aclrtStream *s, const aclrtStreamConfigHandle *cfg) {
 __typeof__(&aclrtCreateStreamV2) fn=(__typeof__(&aclrtCreateStreamV2))originals[10]; uint64_t begin=now_ns();
 aclError rc=fn(s,cfg); uint64_t finish=now_ns();
 record("aclrtCreateStreamV2",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(cfg),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateStreamV2_3(aclrtStream *s, const aclrtStreamConfigHandle *cfg) {
 __typeof__(&aclrtCreateStreamV2) fn=(__typeof__(&aclrtCreateStreamV2))originals[11]; uint64_t begin=now_ns();
 aclError rc=fn(s,cfg); uint64_t finish=now_ns();
 record("aclrtCreateStreamV2",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(cfg),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStream_0(aclrtStream s) {
 __typeof__(&aclrtDestroyStream) fn=(__typeof__(&aclrtDestroyStream))originals[12]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStream_1(aclrtStream s) {
 __typeof__(&aclrtDestroyStream) fn=(__typeof__(&aclrtDestroyStream))originals[13]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStream_2(aclrtStream s) {
 __typeof__(&aclrtDestroyStream) fn=(__typeof__(&aclrtDestroyStream))originals[14]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStream_3(aclrtStream s) {
 __typeof__(&aclrtDestroyStream) fn=(__typeof__(&aclrtDestroyStream))originals[15]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStreamForce_0(aclrtStream s) {
 __typeof__(&aclrtDestroyStreamForce) fn=(__typeof__(&aclrtDestroyStreamForce))originals[16]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStreamForce",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStreamForce_1(aclrtStream s) {
 __typeof__(&aclrtDestroyStreamForce) fn=(__typeof__(&aclrtDestroyStreamForce))originals[17]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStreamForce",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStreamForce_2(aclrtStream s) {
 __typeof__(&aclrtDestroyStreamForce) fn=(__typeof__(&aclrtDestroyStreamForce))originals[18]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStreamForce",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyStreamForce_3(aclrtStream s) {
 __typeof__(&aclrtDestroyStreamForce) fn=(__typeof__(&aclrtDestroyStreamForce))originals[19]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtDestroyStreamForce",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamGetId_0(aclrtStream s, int32_t *id) {
 __typeof__(&aclrtStreamGetId) fn=(__typeof__(&aclrtStreamGetId))originals[20]; uint64_t begin=now_ns();
 aclError rc=fn(s,id); uint64_t finish=now_ns();
 record("aclrtStreamGetId",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*id:-1),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamGetId_1(aclrtStream s, int32_t *id) {
 __typeof__(&aclrtStreamGetId) fn=(__typeof__(&aclrtStreamGetId))originals[21]; uint64_t begin=now_ns();
 aclError rc=fn(s,id); uint64_t finish=now_ns();
 record("aclrtStreamGetId",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*id:-1),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamGetId_2(aclrtStream s, int32_t *id) {
 __typeof__(&aclrtStreamGetId) fn=(__typeof__(&aclrtStreamGetId))originals[22]; uint64_t begin=now_ns();
 aclError rc=fn(s,id); uint64_t finish=now_ns();
 record("aclrtStreamGetId",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*id:-1),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamGetId_3(aclrtStream s, int32_t *id) {
 __typeof__(&aclrtStreamGetId) fn=(__typeof__(&aclrtStreamGetId))originals[23]; uint64_t begin=now_ns();
 aclError rc=fn(s,id); uint64_t finish=now_ns();
 record("aclrtStreamGetId",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*id:-1),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCtxGetCurrentDefaultStream_0(aclrtStream *s) {
 __typeof__(&aclrtCtxGetCurrentDefaultStream) fn=(__typeof__(&aclrtCtxGetCurrentDefaultStream))originals[24]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCtxGetCurrentDefaultStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCtxGetCurrentDefaultStream_1(aclrtStream *s) {
 __typeof__(&aclrtCtxGetCurrentDefaultStream) fn=(__typeof__(&aclrtCtxGetCurrentDefaultStream))originals[25]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCtxGetCurrentDefaultStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCtxGetCurrentDefaultStream_2(aclrtStream *s) {
 __typeof__(&aclrtCtxGetCurrentDefaultStream) fn=(__typeof__(&aclrtCtxGetCurrentDefaultStream))originals[26]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCtxGetCurrentDefaultStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCtxGetCurrentDefaultStream_3(aclrtStream *s) {
 __typeof__(&aclrtCtxGetCurrentDefaultStream) fn=(__typeof__(&aclrtCtxGetCurrentDefaultStream))originals[27]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtCtxGetCurrentDefaultStream",begin,finish,(uintptr_t)(rc==0?*s:0),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEvent_0(aclrtEvent *e) {
 __typeof__(&aclrtCreateEvent) fn=(__typeof__(&aclrtCreateEvent))originals[28]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtCreateEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEvent_1(aclrtEvent *e) {
 __typeof__(&aclrtCreateEvent) fn=(__typeof__(&aclrtCreateEvent))originals[29]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtCreateEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEvent_2(aclrtEvent *e) {
 __typeof__(&aclrtCreateEvent) fn=(__typeof__(&aclrtCreateEvent))originals[30]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtCreateEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEvent_3(aclrtEvent *e) {
 __typeof__(&aclrtCreateEvent) fn=(__typeof__(&aclrtCreateEvent))originals[31]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtCreateEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventWithFlag_0(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventWithFlag) fn=(__typeof__(&aclrtCreateEventWithFlag))originals[32]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventWithFlag_1(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventWithFlag) fn=(__typeof__(&aclrtCreateEventWithFlag))originals[33]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventWithFlag_2(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventWithFlag) fn=(__typeof__(&aclrtCreateEventWithFlag))originals[34]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventWithFlag_3(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventWithFlag) fn=(__typeof__(&aclrtCreateEventWithFlag))originals[35]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventExWithFlag_0(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventExWithFlag) fn=(__typeof__(&aclrtCreateEventExWithFlag))originals[36]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventExWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventExWithFlag_1(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventExWithFlag) fn=(__typeof__(&aclrtCreateEventExWithFlag))originals[37]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventExWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventExWithFlag_2(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventExWithFlag) fn=(__typeof__(&aclrtCreateEventExWithFlag))originals[38]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventExWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtCreateEventExWithFlag_3(aclrtEvent *e, uint32_t flag) {
 __typeof__(&aclrtCreateEventExWithFlag) fn=(__typeof__(&aclrtCreateEventExWithFlag))originals[39]; uint64_t begin=now_ns();
 aclError rc=fn(e,flag); uint64_t finish=now_ns();
 record("aclrtCreateEventExWithFlag",begin,finish,(uintptr_t)(0),(uintptr_t)(rc==0?*e:0),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyEvent_0(aclrtEvent e) {
 __typeof__(&aclrtDestroyEvent) fn=(__typeof__(&aclrtDestroyEvent))originals[40]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtDestroyEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyEvent_1(aclrtEvent e) {
 __typeof__(&aclrtDestroyEvent) fn=(__typeof__(&aclrtDestroyEvent))originals[41]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtDestroyEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyEvent_2(aclrtEvent e) {
 __typeof__(&aclrtDestroyEvent) fn=(__typeof__(&aclrtDestroyEvent))originals[42]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtDestroyEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtDestroyEvent_3(aclrtEvent e) {
 __typeof__(&aclrtDestroyEvent) fn=(__typeof__(&aclrtDestroyEvent))originals[43]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtDestroyEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtRecordEvent_0(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtRecordEvent) fn=(__typeof__(&aclrtRecordEvent))originals[44]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtRecordEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtRecordEvent_1(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtRecordEvent) fn=(__typeof__(&aclrtRecordEvent))originals[45]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtRecordEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtRecordEvent_2(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtRecordEvent) fn=(__typeof__(&aclrtRecordEvent))originals[46]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtRecordEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtRecordEvent_3(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtRecordEvent) fn=(__typeof__(&aclrtRecordEvent))originals[47]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtRecordEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtResetEvent_0(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtResetEvent) fn=(__typeof__(&aclrtResetEvent))originals[48]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtResetEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtResetEvent_1(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtResetEvent) fn=(__typeof__(&aclrtResetEvent))originals[49]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtResetEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtResetEvent_2(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtResetEvent) fn=(__typeof__(&aclrtResetEvent))originals[50]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtResetEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtResetEvent_3(aclrtEvent e, aclrtStream s) {
 __typeof__(&aclrtResetEvent) fn=(__typeof__(&aclrtResetEvent))originals[51]; uint64_t begin=now_ns();
 aclError rc=fn(e,s); uint64_t finish=now_ns();
 record("aclrtResetEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEvent_0(aclrtStream s, aclrtEvent e) {
 __typeof__(&aclrtStreamWaitEvent) fn=(__typeof__(&aclrtStreamWaitEvent))originals[52]; uint64_t begin=now_ns();
 aclError rc=fn(s,e); uint64_t finish=now_ns();
 record("aclrtStreamWaitEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEvent_1(aclrtStream s, aclrtEvent e) {
 __typeof__(&aclrtStreamWaitEvent) fn=(__typeof__(&aclrtStreamWaitEvent))originals[53]; uint64_t begin=now_ns();
 aclError rc=fn(s,e); uint64_t finish=now_ns();
 record("aclrtStreamWaitEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEvent_2(aclrtStream s, aclrtEvent e) {
 __typeof__(&aclrtStreamWaitEvent) fn=(__typeof__(&aclrtStreamWaitEvent))originals[54]; uint64_t begin=now_ns();
 aclError rc=fn(s,e); uint64_t finish=now_ns();
 record("aclrtStreamWaitEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEvent_3(aclrtStream s, aclrtEvent e) {
 __typeof__(&aclrtStreamWaitEvent) fn=(__typeof__(&aclrtStreamWaitEvent))originals[55]; uint64_t begin=now_ns();
 aclError rc=fn(s,e); uint64_t finish=now_ns();
 record("aclrtStreamWaitEvent",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEventWithTimeout_0(aclrtStream s, aclrtEvent e, int32_t timeout) {
 __typeof__(&aclrtStreamWaitEventWithTimeout) fn=(__typeof__(&aclrtStreamWaitEventWithTimeout))originals[56]; uint64_t begin=now_ns();
 aclError rc=fn(s,e,timeout); uint64_t finish=now_ns();
 record("aclrtStreamWaitEventWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEventWithTimeout_1(aclrtStream s, aclrtEvent e, int32_t timeout) {
 __typeof__(&aclrtStreamWaitEventWithTimeout) fn=(__typeof__(&aclrtStreamWaitEventWithTimeout))originals[57]; uint64_t begin=now_ns();
 aclError rc=fn(s,e,timeout); uint64_t finish=now_ns();
 record("aclrtStreamWaitEventWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEventWithTimeout_2(aclrtStream s, aclrtEvent e, int32_t timeout) {
 __typeof__(&aclrtStreamWaitEventWithTimeout) fn=(__typeof__(&aclrtStreamWaitEventWithTimeout))originals[58]; uint64_t begin=now_ns();
 aclError rc=fn(s,e,timeout); uint64_t finish=now_ns();
 record("aclrtStreamWaitEventWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtStreamWaitEventWithTimeout_3(aclrtStream s, aclrtEvent e, int32_t timeout) {
 __typeof__(&aclrtStreamWaitEventWithTimeout) fn=(__typeof__(&aclrtStreamWaitEventWithTimeout))originals[59]; uint64_t begin=now_ns();
 aclError rc=fn(s,e,timeout); uint64_t finish=now_ns();
 record("aclrtStreamWaitEventWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(e),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStream_0(aclrtStream s) {
 __typeof__(&aclrtSynchronizeStream) fn=(__typeof__(&aclrtSynchronizeStream))originals[60]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtSynchronizeStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStream_1(aclrtStream s) {
 __typeof__(&aclrtSynchronizeStream) fn=(__typeof__(&aclrtSynchronizeStream))originals[61]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtSynchronizeStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStream_2(aclrtStream s) {
 __typeof__(&aclrtSynchronizeStream) fn=(__typeof__(&aclrtSynchronizeStream))originals[62]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtSynchronizeStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStream_3(aclrtStream s) {
 __typeof__(&aclrtSynchronizeStream) fn=(__typeof__(&aclrtSynchronizeStream))originals[63]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclrtSynchronizeStream",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStreamWithTimeout_0(aclrtStream s, int32_t timeout) {
 __typeof__(&aclrtSynchronizeStreamWithTimeout) fn=(__typeof__(&aclrtSynchronizeStreamWithTimeout))originals[64]; uint64_t begin=now_ns();
 aclError rc=fn(s,timeout); uint64_t finish=now_ns();
 record("aclrtSynchronizeStreamWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStreamWithTimeout_1(aclrtStream s, int32_t timeout) {
 __typeof__(&aclrtSynchronizeStreamWithTimeout) fn=(__typeof__(&aclrtSynchronizeStreamWithTimeout))originals[65]; uint64_t begin=now_ns();
 aclError rc=fn(s,timeout); uint64_t finish=now_ns();
 record("aclrtSynchronizeStreamWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStreamWithTimeout_2(aclrtStream s, int32_t timeout) {
 __typeof__(&aclrtSynchronizeStreamWithTimeout) fn=(__typeof__(&aclrtSynchronizeStreamWithTimeout))originals[66]; uint64_t begin=now_ns();
 aclError rc=fn(s,timeout); uint64_t finish=now_ns();
 record("aclrtSynchronizeStreamWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeStreamWithTimeout_3(aclrtStream s, int32_t timeout) {
 __typeof__(&aclrtSynchronizeStreamWithTimeout) fn=(__typeof__(&aclrtSynchronizeStreamWithTimeout))originals[67]; uint64_t begin=now_ns();
 aclError rc=fn(s,timeout); uint64_t finish=now_ns();
 record("aclrtSynchronizeStreamWithTimeout",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(timeout),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeEvent_0(aclrtEvent e) {
 __typeof__(&aclrtSynchronizeEvent) fn=(__typeof__(&aclrtSynchronizeEvent))originals[68]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtSynchronizeEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeEvent_1(aclrtEvent e) {
 __typeof__(&aclrtSynchronizeEvent) fn=(__typeof__(&aclrtSynchronizeEvent))originals[69]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtSynchronizeEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeEvent_2(aclrtEvent e) {
 __typeof__(&aclrtSynchronizeEvent) fn=(__typeof__(&aclrtSynchronizeEvent))originals[70]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtSynchronizeEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSynchronizeEvent_3(aclrtEvent e) {
 __typeof__(&aclrtSynchronizeEvent) fn=(__typeof__(&aclrtSynchronizeEvent))originals[71]; uint64_t begin=now_ns();
 aclError rc=fn(e); uint64_t finish=now_ns();
 record("aclrtSynchronizeEvent",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtQueryEventStatus_0(aclrtEvent e, aclrtEventRecordedStatus *status) {
 __typeof__(&aclrtQueryEventStatus) fn=(__typeof__(&aclrtQueryEventStatus))originals[72]; uint64_t begin=now_ns();
 aclError rc=fn(e,status); uint64_t finish=now_ns();
 record("aclrtQueryEventStatus",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(rc==0?*status:-1),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtQueryEventStatus_1(aclrtEvent e, aclrtEventRecordedStatus *status) {
 __typeof__(&aclrtQueryEventStatus) fn=(__typeof__(&aclrtQueryEventStatus))originals[73]; uint64_t begin=now_ns();
 aclError rc=fn(e,status); uint64_t finish=now_ns();
 record("aclrtQueryEventStatus",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(rc==0?*status:-1),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtQueryEventStatus_2(aclrtEvent e, aclrtEventRecordedStatus *status) {
 __typeof__(&aclrtQueryEventStatus) fn=(__typeof__(&aclrtQueryEventStatus))originals[74]; uint64_t begin=now_ns();
 aclError rc=fn(e,status); uint64_t finish=now_ns();
 record("aclrtQueryEventStatus",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(rc==0?*status:-1),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtQueryEventStatus_3(aclrtEvent e, aclrtEventRecordedStatus *status) {
 __typeof__(&aclrtQueryEventStatus) fn=(__typeof__(&aclrtQueryEventStatus))originals[75]; uint64_t begin=now_ns();
 aclError rc=fn(e,status); uint64_t finish=now_ns();
 record("aclrtQueryEventStatus",begin,finish,(uintptr_t)(0),(uintptr_t)(e),(uint64_t)(rc==0?*status:-1),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSetStreamResLimit_0(aclrtStream s, aclrtDevResLimitType type, uint32_t value) {
 __typeof__(&aclrtSetStreamResLimit) fn=(__typeof__(&aclrtSetStreamResLimit))originals[76]; uint64_t begin=now_ns();
 aclError rc=fn(s,type,value); uint64_t finish=now_ns();
 record("aclrtSetStreamResLimit",begin,finish,(uintptr_t)(s),(uintptr_t)(type),(uint64_t)(value),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSetStreamResLimit_1(aclrtStream s, aclrtDevResLimitType type, uint32_t value) {
 __typeof__(&aclrtSetStreamResLimit) fn=(__typeof__(&aclrtSetStreamResLimit))originals[77]; uint64_t begin=now_ns();
 aclError rc=fn(s,type,value); uint64_t finish=now_ns();
 record("aclrtSetStreamResLimit",begin,finish,(uintptr_t)(s),(uintptr_t)(type),(uint64_t)(value),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSetStreamResLimit_2(aclrtStream s, aclrtDevResLimitType type, uint32_t value) {
 __typeof__(&aclrtSetStreamResLimit) fn=(__typeof__(&aclrtSetStreamResLimit))originals[78]; uint64_t begin=now_ns();
 aclError rc=fn(s,type,value); uint64_t finish=now_ns();
 record("aclrtSetStreamResLimit",begin,finish,(uintptr_t)(s),(uintptr_t)(type),(uint64_t)(value),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclrtSetStreamResLimit_3(aclrtStream s, aclrtDevResLimitType type, uint32_t value) {
 __typeof__(&aclrtSetStreamResLimit) fn=(__typeof__(&aclrtSetStreamResLimit))originals[79]; uint64_t begin=now_ns();
 aclError rc=fn(s,type,value); uint64_t finish=now_ns();
 record("aclrtSetStreamResLimit",begin,finish,(uintptr_t)(s),(uintptr_t)(type),(uint64_t)(value),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureBegin_0(aclrtStream s, aclmdlRICaptureMode mode) {
 __typeof__(&aclmdlRICaptureBegin) fn=(__typeof__(&aclmdlRICaptureBegin))originals[80]; uint64_t begin=now_ns();
 aclError rc=fn(s,mode); uint64_t finish=now_ns();
 record("aclmdlRICaptureBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(mode),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureBegin_1(aclrtStream s, aclmdlRICaptureMode mode) {
 __typeof__(&aclmdlRICaptureBegin) fn=(__typeof__(&aclmdlRICaptureBegin))originals[81]; uint64_t begin=now_ns();
 aclError rc=fn(s,mode); uint64_t finish=now_ns();
 record("aclmdlRICaptureBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(mode),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureBegin_2(aclrtStream s, aclmdlRICaptureMode mode) {
 __typeof__(&aclmdlRICaptureBegin) fn=(__typeof__(&aclmdlRICaptureBegin))originals[82]; uint64_t begin=now_ns();
 aclError rc=fn(s,mode); uint64_t finish=now_ns();
 record("aclmdlRICaptureBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(mode),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureBegin_3(aclrtStream s, aclmdlRICaptureMode mode) {
 __typeof__(&aclmdlRICaptureBegin) fn=(__typeof__(&aclmdlRICaptureBegin))originals[83]; uint64_t begin=now_ns();
 aclError rc=fn(s,mode); uint64_t finish=now_ns();
 record("aclmdlRICaptureBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(mode),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureEnd_0(aclrtStream s, aclmdlRI *m) {
 __typeof__(&aclmdlRICaptureEnd) fn=(__typeof__(&aclmdlRICaptureEnd))originals[84]; uint64_t begin=now_ns();
 aclError rc=fn(s,m); uint64_t finish=now_ns();
 record("aclmdlRICaptureEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*m:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureEnd_1(aclrtStream s, aclmdlRI *m) {
 __typeof__(&aclmdlRICaptureEnd) fn=(__typeof__(&aclmdlRICaptureEnd))originals[85]; uint64_t begin=now_ns();
 aclError rc=fn(s,m); uint64_t finish=now_ns();
 record("aclmdlRICaptureEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*m:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureEnd_2(aclrtStream s, aclmdlRI *m) {
 __typeof__(&aclmdlRICaptureEnd) fn=(__typeof__(&aclmdlRICaptureEnd))originals[86]; uint64_t begin=now_ns();
 aclError rc=fn(s,m); uint64_t finish=now_ns();
 record("aclmdlRICaptureEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*m:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureEnd_3(aclrtStream s, aclmdlRI *m) {
 __typeof__(&aclmdlRICaptureEnd) fn=(__typeof__(&aclmdlRICaptureEnd))originals[87]; uint64_t begin=now_ns();
 aclError rc=fn(s,m); uint64_t finish=now_ns();
 record("aclmdlRICaptureEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*m:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIExecuteAsync_0(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIExecuteAsync) fn=(__typeof__(&aclmdlRIExecuteAsync))originals[88]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIExecuteAsync",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIExecuteAsync_1(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIExecuteAsync) fn=(__typeof__(&aclmdlRIExecuteAsync))originals[89]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIExecuteAsync",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIExecuteAsync_2(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIExecuteAsync) fn=(__typeof__(&aclmdlRIExecuteAsync))originals[90]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIExecuteAsync",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIExecuteAsync_3(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIExecuteAsync) fn=(__typeof__(&aclmdlRIExecuteAsync))originals[91]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIExecuteAsync",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIDestroy_0(aclmdlRI m) {
 __typeof__(&aclmdlRIDestroy) fn=(__typeof__(&aclmdlRIDestroy))originals[92]; uint64_t begin=now_ns();
 aclError rc=fn(m); uint64_t finish=now_ns();
 record("aclmdlRIDestroy",begin,finish,(uintptr_t)(0),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIDestroy_1(aclmdlRI m) {
 __typeof__(&aclmdlRIDestroy) fn=(__typeof__(&aclmdlRIDestroy))originals[93]; uint64_t begin=now_ns();
 aclError rc=fn(m); uint64_t finish=now_ns();
 record("aclmdlRIDestroy",begin,finish,(uintptr_t)(0),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIDestroy_2(aclmdlRI m) {
 __typeof__(&aclmdlRIDestroy) fn=(__typeof__(&aclmdlRIDestroy))originals[94]; uint64_t begin=now_ns();
 aclError rc=fn(m); uint64_t finish=now_ns();
 record("aclmdlRIDestroy",begin,finish,(uintptr_t)(0),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIDestroy_3(aclmdlRI m) {
 __typeof__(&aclmdlRIDestroy) fn=(__typeof__(&aclmdlRIDestroy))originals[95]; uint64_t begin=now_ns();
 aclError rc=fn(m); uint64_t finish=now_ns();
 record("aclmdlRIDestroy",begin,finish,(uintptr_t)(0),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIBindStream_0(aclmdlRI m, aclrtStream s, uint32_t flag) {
 __typeof__(&aclmdlRIBindStream) fn=(__typeof__(&aclmdlRIBindStream))originals[96]; uint64_t begin=now_ns();
 aclError rc=fn(m,s,flag); uint64_t finish=now_ns();
 record("aclmdlRIBindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIBindStream_1(aclmdlRI m, aclrtStream s, uint32_t flag) {
 __typeof__(&aclmdlRIBindStream) fn=(__typeof__(&aclmdlRIBindStream))originals[97]; uint64_t begin=now_ns();
 aclError rc=fn(m,s,flag); uint64_t finish=now_ns();
 record("aclmdlRIBindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIBindStream_2(aclmdlRI m, aclrtStream s, uint32_t flag) {
 __typeof__(&aclmdlRIBindStream) fn=(__typeof__(&aclmdlRIBindStream))originals[98]; uint64_t begin=now_ns();
 aclError rc=fn(m,s,flag); uint64_t finish=now_ns();
 record("aclmdlRIBindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIBindStream_3(aclmdlRI m, aclrtStream s, uint32_t flag) {
 __typeof__(&aclmdlRIBindStream) fn=(__typeof__(&aclmdlRIBindStream))originals[99]; uint64_t begin=now_ns();
 aclError rc=fn(m,s,flag); uint64_t finish=now_ns();
 record("aclmdlRIBindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(flag),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIUnbindStream_0(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIUnbindStream) fn=(__typeof__(&aclmdlRIUnbindStream))originals[100]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIUnbindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIUnbindStream_1(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIUnbindStream) fn=(__typeof__(&aclmdlRIUnbindStream))originals[101]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIUnbindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIUnbindStream_2(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIUnbindStream) fn=(__typeof__(&aclmdlRIUnbindStream))originals[102]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIUnbindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRIUnbindStream_3(aclmdlRI m, aclrtStream s) {
 __typeof__(&aclmdlRIUnbindStream) fn=(__typeof__(&aclmdlRIUnbindStream))originals[103]; uint64_t begin=now_ns();
 aclError rc=fn(m,s); uint64_t finish=now_ns();
 record("aclmdlRIUnbindStream",begin,finish,(uintptr_t)(s),(uintptr_t)(m),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpBegin_0(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskGrpBegin) fn=(__typeof__(&aclmdlRICaptureTaskGrpBegin))originals[104]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpBegin_1(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskGrpBegin) fn=(__typeof__(&aclmdlRICaptureTaskGrpBegin))originals[105]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpBegin_2(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskGrpBegin) fn=(__typeof__(&aclmdlRICaptureTaskGrpBegin))originals[106]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpBegin_3(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskGrpBegin) fn=(__typeof__(&aclmdlRICaptureTaskGrpBegin))originals[107]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpEnd_0(aclrtStream s, aclrtTaskGrp *h) {
 __typeof__(&aclmdlRICaptureTaskGrpEnd) fn=(__typeof__(&aclmdlRICaptureTaskGrpEnd))originals[108]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*h:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpEnd_1(aclrtStream s, aclrtTaskGrp *h) {
 __typeof__(&aclmdlRICaptureTaskGrpEnd) fn=(__typeof__(&aclmdlRICaptureTaskGrpEnd))originals[109]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*h:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpEnd_2(aclrtStream s, aclrtTaskGrp *h) {
 __typeof__(&aclmdlRICaptureTaskGrpEnd) fn=(__typeof__(&aclmdlRICaptureTaskGrpEnd))originals[110]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*h:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskGrpEnd_3(aclrtStream s, aclrtTaskGrp *h) {
 __typeof__(&aclmdlRICaptureTaskGrpEnd) fn=(__typeof__(&aclmdlRICaptureTaskGrpEnd))originals[111]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskGrpEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(rc==0?*h:0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateBegin_0(aclrtStream s, aclrtTaskGrp h) {
 __typeof__(&aclmdlRICaptureTaskUpdateBegin) fn=(__typeof__(&aclmdlRICaptureTaskUpdateBegin))originals[112]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(h),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateBegin_1(aclrtStream s, aclrtTaskGrp h) {
 __typeof__(&aclmdlRICaptureTaskUpdateBegin) fn=(__typeof__(&aclmdlRICaptureTaskUpdateBegin))originals[113]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(h),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateBegin_2(aclrtStream s, aclrtTaskGrp h) {
 __typeof__(&aclmdlRICaptureTaskUpdateBegin) fn=(__typeof__(&aclmdlRICaptureTaskUpdateBegin))originals[114]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(h),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateBegin_3(aclrtStream s, aclrtTaskGrp h) {
 __typeof__(&aclmdlRICaptureTaskUpdateBegin) fn=(__typeof__(&aclmdlRICaptureTaskUpdateBegin))originals[115]; uint64_t begin=now_ns();
 aclError rc=fn(s,h); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateBegin",begin,finish,(uintptr_t)(s),(uintptr_t)(h),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateEnd_0(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskUpdateEnd) fn=(__typeof__(&aclmdlRICaptureTaskUpdateEnd))originals[116]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateEnd_1(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskUpdateEnd) fn=(__typeof__(&aclmdlRICaptureTaskUpdateEnd))originals[117]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateEnd_2(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskUpdateEnd) fn=(__typeof__(&aclmdlRICaptureTaskUpdateEnd))originals[118]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
static aclError p22_aclmdlRICaptureTaskUpdateEnd_3(aclrtStream s) {
 __typeof__(&aclmdlRICaptureTaskUpdateEnd) fn=(__typeof__(&aclmdlRICaptureTaskUpdateEnd))originals[119]; uint64_t begin=now_ns();
 aclError rc=fn(s); uint64_t finish=now_ns();
 record("aclmdlRICaptureTaskUpdateEnd",begin,finish,(uintptr_t)(s),(uintptr_t)(0),(uint64_t)(0),rc,__builtin_return_address(0)); return rc;
}
unsigned int la_version(unsigned int version) {(void)version;return LAV_CURRENT;}
unsigned int la_objopen(struct link_map *map,Lmid_t lmid,uintptr_t *cookie) {(void)lmid;*cookie=(uintptr_t)map;return LA_FLG_BINDTO|LA_FLG_BINDFROM;}
uintptr_t la_symbind64(Elf64_Sym *symbol,unsigned int index,uintptr_t *ref,uintptr_t *def,unsigned int *flags,const char *name) {
    (void)index;(void)def;(void)flags;
    const char *requester=((struct link_map*)*ref)->l_name;
    if(strstr(requester,"native_stream_trace")) return symbol->st_value;
    if(!strstr(requester,"torch_npu") && !strstr(requester,"libatb") && !strstr(requester,"libopapi")) return symbol->st_value;
    static const char *names[]={"aclrtCreateStream","aclrtCreateStreamWithConfig","aclrtCreateStreamV2","aclrtDestroyStream","aclrtDestroyStreamForce","aclrtStreamGetId","aclrtCtxGetCurrentDefaultStream","aclrtCreateEvent","aclrtCreateEventWithFlag","aclrtCreateEventExWithFlag","aclrtDestroyEvent","aclrtRecordEvent","aclrtResetEvent","aclrtStreamWaitEvent","aclrtStreamWaitEventWithTimeout","aclrtSynchronizeStream","aclrtSynchronizeStreamWithTimeout","aclrtSynchronizeEvent","aclrtQueryEventStatus","aclrtSetStreamResLimit","aclmdlRICaptureBegin","aclmdlRICaptureEnd","aclmdlRIExecuteAsync","aclmdlRIDestroy","aclmdlRIBindStream","aclmdlRIUnbindStream","aclmdlRICaptureTaskGrpBegin","aclmdlRICaptureTaskGrpEnd","aclmdlRICaptureTaskUpdateBegin","aclmdlRICaptureTaskUpdateEnd"};void *functions[][4]={{(void*)p22_aclrtCreateStream_0,(void*)p22_aclrtCreateStream_1,(void*)p22_aclrtCreateStream_2,(void*)p22_aclrtCreateStream_3},{(void*)p22_aclrtCreateStreamWithConfig_0,(void*)p22_aclrtCreateStreamWithConfig_1,(void*)p22_aclrtCreateStreamWithConfig_2,(void*)p22_aclrtCreateStreamWithConfig_3},{(void*)p22_aclrtCreateStreamV2_0,(void*)p22_aclrtCreateStreamV2_1,(void*)p22_aclrtCreateStreamV2_2,(void*)p22_aclrtCreateStreamV2_3},{(void*)p22_aclrtDestroyStream_0,(void*)p22_aclrtDestroyStream_1,(void*)p22_aclrtDestroyStream_2,(void*)p22_aclrtDestroyStream_3},{(void*)p22_aclrtDestroyStreamForce_0,(void*)p22_aclrtDestroyStreamForce_1,(void*)p22_aclrtDestroyStreamForce_2,(void*)p22_aclrtDestroyStreamForce_3},{(void*)p22_aclrtStreamGetId_0,(void*)p22_aclrtStreamGetId_1,(void*)p22_aclrtStreamGetId_2,(void*)p22_aclrtStreamGetId_3},{(void*)p22_aclrtCtxGetCurrentDefaultStream_0,(void*)p22_aclrtCtxGetCurrentDefaultStream_1,(void*)p22_aclrtCtxGetCurrentDefaultStream_2,(void*)p22_aclrtCtxGetCurrentDefaultStream_3},{(void*)p22_aclrtCreateEvent_0,(void*)p22_aclrtCreateEvent_1,(void*)p22_aclrtCreateEvent_2,(void*)p22_aclrtCreateEvent_3},{(void*)p22_aclrtCreateEventWithFlag_0,(void*)p22_aclrtCreateEventWithFlag_1,(void*)p22_aclrtCreateEventWithFlag_2,(void*)p22_aclrtCreateEventWithFlag_3},{(void*)p22_aclrtCreateEventExWithFlag_0,(void*)p22_aclrtCreateEventExWithFlag_1,(void*)p22_aclrtCreateEventExWithFlag_2,(void*)p22_aclrtCreateEventExWithFlag_3},{(void*)p22_aclrtDestroyEvent_0,(void*)p22_aclrtDestroyEvent_1,(void*)p22_aclrtDestroyEvent_2,(void*)p22_aclrtDestroyEvent_3},{(void*)p22_aclrtRecordEvent_0,(void*)p22_aclrtRecordEvent_1,(void*)p22_aclrtRecordEvent_2,(void*)p22_aclrtRecordEvent_3},{(void*)p22_aclrtResetEvent_0,(void*)p22_aclrtResetEvent_1,(void*)p22_aclrtResetEvent_2,(void*)p22_aclrtResetEvent_3},{(void*)p22_aclrtStreamWaitEvent_0,(void*)p22_aclrtStreamWaitEvent_1,(void*)p22_aclrtStreamWaitEvent_2,(void*)p22_aclrtStreamWaitEvent_3},{(void*)p22_aclrtStreamWaitEventWithTimeout_0,(void*)p22_aclrtStreamWaitEventWithTimeout_1,(void*)p22_aclrtStreamWaitEventWithTimeout_2,(void*)p22_aclrtStreamWaitEventWithTimeout_3},{(void*)p22_aclrtSynchronizeStream_0,(void*)p22_aclrtSynchronizeStream_1,(void*)p22_aclrtSynchronizeStream_2,(void*)p22_aclrtSynchronizeStream_3},{(void*)p22_aclrtSynchronizeStreamWithTimeout_0,(void*)p22_aclrtSynchronizeStreamWithTimeout_1,(void*)p22_aclrtSynchronizeStreamWithTimeout_2,(void*)p22_aclrtSynchronizeStreamWithTimeout_3},{(void*)p22_aclrtSynchronizeEvent_0,(void*)p22_aclrtSynchronizeEvent_1,(void*)p22_aclrtSynchronizeEvent_2,(void*)p22_aclrtSynchronizeEvent_3},{(void*)p22_aclrtQueryEventStatus_0,(void*)p22_aclrtQueryEventStatus_1,(void*)p22_aclrtQueryEventStatus_2,(void*)p22_aclrtQueryEventStatus_3},{(void*)p22_aclrtSetStreamResLimit_0,(void*)p22_aclrtSetStreamResLimit_1,(void*)p22_aclrtSetStreamResLimit_2,(void*)p22_aclrtSetStreamResLimit_3},{(void*)p22_aclmdlRICaptureBegin_0,(void*)p22_aclmdlRICaptureBegin_1,(void*)p22_aclmdlRICaptureBegin_2,(void*)p22_aclmdlRICaptureBegin_3},{(void*)p22_aclmdlRICaptureEnd_0,(void*)p22_aclmdlRICaptureEnd_1,(void*)p22_aclmdlRICaptureEnd_2,(void*)p22_aclmdlRICaptureEnd_3},{(void*)p22_aclmdlRIExecuteAsync_0,(void*)p22_aclmdlRIExecuteAsync_1,(void*)p22_aclmdlRIExecuteAsync_2,(void*)p22_aclmdlRIExecuteAsync_3},{(void*)p22_aclmdlRIDestroy_0,(void*)p22_aclmdlRIDestroy_1,(void*)p22_aclmdlRIDestroy_2,(void*)p22_aclmdlRIDestroy_3},{(void*)p22_aclmdlRIBindStream_0,(void*)p22_aclmdlRIBindStream_1,(void*)p22_aclmdlRIBindStream_2,(void*)p22_aclmdlRIBindStream_3},{(void*)p22_aclmdlRIUnbindStream_0,(void*)p22_aclmdlRIUnbindStream_1,(void*)p22_aclmdlRIUnbindStream_2,(void*)p22_aclmdlRIUnbindStream_3},{(void*)p22_aclmdlRICaptureTaskGrpBegin_0,(void*)p22_aclmdlRICaptureTaskGrpBegin_1,(void*)p22_aclmdlRICaptureTaskGrpBegin_2,(void*)p22_aclmdlRICaptureTaskGrpBegin_3},{(void*)p22_aclmdlRICaptureTaskGrpEnd_0,(void*)p22_aclmdlRICaptureTaskGrpEnd_1,(void*)p22_aclmdlRICaptureTaskGrpEnd_2,(void*)p22_aclmdlRICaptureTaskGrpEnd_3},{(void*)p22_aclmdlRICaptureTaskUpdateBegin_0,(void*)p22_aclmdlRICaptureTaskUpdateBegin_1,(void*)p22_aclmdlRICaptureTaskUpdateBegin_2,(void*)p22_aclmdlRICaptureTaskUpdateBegin_3},{(void*)p22_aclmdlRICaptureTaskUpdateEnd_0,(void*)p22_aclmdlRICaptureTaskUpdateEnd_1,(void*)p22_aclmdlRICaptureTaskUpdateEnd_2,(void*)p22_aclmdlRICaptureTaskUpdateEnd_3}};
    for(size_t i=0;i<sizeof(names)/sizeof(names[0]);i++) {
        if(strcmp(name,names[i]))continue;
        for(size_t v=0;v<4;v++) {
            uintptr_t old=__atomic_load_n(&originals[i*4+v],__ATOMIC_ACQUIRE);
            if(old==symbol->st_value)return (uintptr_t)functions[i][v];
            if(!old && __atomic_compare_exchange_n(&originals[i*4+v],&old,symbol->st_value,0,__ATOMIC_RELEASE,__ATOMIC_ACQUIRE)) return (uintptr_t)functions[i][v];
            if(old==symbol->st_value)return (uintptr_t)functions[i][v];
        }
        fprintf(stderr,"P22 too many implementations for %s from %s\n",name,requester);_exit(125);
    }
    return symbol->st_value;
}
