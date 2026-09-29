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
/*WRAPPERS*/
unsigned int la_version(unsigned int version) {(void)version;return LAV_CURRENT;}
unsigned int la_objopen(struct link_map *map,Lmid_t lmid,uintptr_t *cookie) {(void)lmid;*cookie=(uintptr_t)map;return LA_FLG_BINDTO|LA_FLG_BINDFROM;}
uintptr_t la_symbind64(Elf64_Sym *symbol,unsigned int index,uintptr_t *ref,uintptr_t *def,unsigned int *flags,const char *name) {
    (void)index;(void)def;(void)flags;
    const char *requester=((struct link_map*)*ref)->l_name;
    if(strstr(requester,"native_stream_trace")) return symbol->st_value;
    if(!strstr(requester,"torch_npu") && !strstr(requester,"libatb") && !strstr(requester,"libopapi")) return symbol->st_value;
    static const char *names[]={/*NAMES*/};void *functions[][4]={/*FUNCTIONS*/};
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
