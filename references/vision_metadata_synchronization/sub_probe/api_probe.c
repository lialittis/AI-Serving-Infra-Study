/* Diagnostic-only dlsym interposition. Original arguments/results are preserved. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef struct { unsigned scope,kind; uint64_t begin,end,tid,u[12]; int result; char path[512]; } Row;
static Row rows[4096];
static _Atomic unsigned used,scope;
static void *(*real_dlsym)(void *, const char *);
static void *resolve(void *h,const char *name) {
    if (!real_dlsym) real_dlsym=dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
    if (!real_dlsym) abort();
    return real_dlsym(h,name);
}
static uint64_t now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ULL+t.tv_nsec; }
static Row begin(unsigned k) { return (Row){.scope=atomic_load(&scope),.kind=k,.begin=now(),.tid=syscall(SYS_gettid)}; }
static void record(Row r) { if (!r.scope) return;r.end=now();unsigned i=atomic_fetch_add(&used,1);if(i>=4096)abort();rows[i]=r; }
void sub_probe_scope(unsigned s) { atomic_store(&scope,s); }

typedef void *(*Create)(const int64_t *,uint64_t,int,const int64_t *,int64_t,int,const int64_t *,uint64_t,void *);
typedef int (*Workspace)(const void *,const void *,const void *,void *,uint64_t *,void **);
typedef int (*Launch)(void *,uint64_t,void *,void *);
typedef void *(*Cache)(const uint8_t *,uint64_t,uint64_t *);
typedef void *(*CacheV1)(uint64_t,uint64_t *);
typedef void (*Address)(void *);
static Create real_create;static Workspace real_workspace;static Launch real_launch;static Cache real_cache;
static CacheV1 real_cache_v1;static Address real_address;
static void *tap_create(const int64_t *shape,uint64_t n,int dtype,const int64_t *stride,int64_t offset,
                       int format,const int64_t *storage,uint64_t sn,void *data) {
    Row r=begin(1);void *out=real_create(shape,n,dtype,stride,offset,format,storage,sn,data);
    r.u[0]=(uintptr_t)out;r.u[1]=(uintptr_t)data;r.u[2]=offset;r.u[3]=n;
    r.u[4]=n?shape[0]:0;r.u[5]=n?stride[0]:0;r.u[6]=dtype;r.u[7]=format;r.u[8]=sn;r.u[9]=sn?storage[0]:0;
    record(r);return out;
}
static int tap_workspace(const void *a,const void *b,const void *alpha,void *out,uint64_t *bytes,void **executor) {
    Row r=begin(2);r.u[0]=(uintptr_t)a;r.u[1]=(uintptr_t)b;r.u[2]=(uintptr_t)alpha;r.u[3]=(uintptr_t)out;
    r.result=real_workspace(a,b,alpha,out,bytes,executor);r.u[4]=*bytes;r.u[5]=(uintptr_t)*executor;record(r);return r.result;
}
static int tap_launch(void *workspace,uint64_t bytes,void *executor,void *stream) {
    Row r=begin(3);r.u[0]=(uintptr_t)workspace;r.u[1]=bytes;r.u[2]=(uintptr_t)executor;r.u[3]=(uintptr_t)stream;
    r.result=real_launch(workspace,bytes,executor,stream);record(r);return r.result;
}
static void *tap_cache(const uint8_t *key,uint64_t len,uint64_t *bytes) {
    Row r=begin(6);void *e=real_cache(key,len,bytes);r.u[0]=(uintptr_t)e;r.u[1]=*bytes;r.u[2]=len;record(r);return e;
}
static void *tap_cache_v1(uint64_t key,uint64_t *bytes) {
    Row r=begin(7);void *e=real_cache_v1(key,bytes);r.u[0]=(uintptr_t)e;r.u[1]=*bytes;r.u[2]=key;record(r);return e;
}
static void tap_address(void *p) {
    Row r=begin(8);r.u[0]=(uintptr_t)p;real_address(p);record(r);
}
void *dlsym(void *h,const char *name) {
    void *p=resolve(h,name),*replacement=NULL;
    if (!strcmp(name,"aclCreateTensor")) {real_create=(Create)p;replacement=tap_create;}
    if (!strcmp(name,"aclnnSubGetWorkspaceSize")) {real_workspace=(Workspace)p;replacement=tap_workspace;}
    if (!strcmp(name,"aclnnSub")) {real_launch=(Launch)p;replacement=tap_launch;}
    if (!strcmp(name,"PTAFindExecCache")) {real_cache=(Cache)p;replacement=tap_cache;}
    if (!strcmp(name,"PTAGetExecCache")) {real_cache_v1=(CacheV1)p;replacement=tap_cache_v1;}
    if (!strcmp(name,"AddTensorAddrToCachedList")) {real_address=(Address)p;replacement=tap_address;}
    if (p && replacement) {
        Row r=begin(5);Dl_info info={0};dladdr(p,&info);r.u[0]=(uintptr_t)p;
        snprintf(r.path,sizeof(r.path),"%s:%s",name,info.dli_fname?info.dli_fname:"");record(r);return replacement;
    }
    return p;
}
static void file_record(const char *p,int result) {
    if (!p || !strstr(p,"/Ascend/") || !strcasestr(p,"sub")) return;
    Row r=begin(4);r.result=result;snprintf(r.path,sizeof(r.path),"%s",p);record(r);
}
int open(const char *p,int flags,...) {
    typedef int(*Fn)(const char *,int,...);static Fn next;if(!next)next=(Fn)resolve(RTLD_NEXT,"open");
    mode_t mode=0;if(flags&O_CREAT){va_list a;va_start(a,flags);mode=va_arg(a,int);va_end(a);}
    int r=next(p,flags,mode);file_record(p,r);return r;
}
int open64(const char *p,int flags,...) {
    typedef int(*Fn)(const char *,int,...);static Fn next;if(!next)next=(Fn)resolve(RTLD_NEXT,"open64");
    mode_t mode=0;if(flags&O_CREAT){va_list a;va_start(a,flags);mode=va_arg(a,int);va_end(a);}
    int r=next(p,flags,mode);file_record(p,r);return r;
}
FILE *fopen(const char *p,const char *mode) {
    typedef FILE *(*Fn)(const char *,const char *);static Fn next;if(!next)next=(Fn)resolve(RTLD_NEXT,"fopen");
    FILE *r=next(p,mode);file_record(p,r?0:-1);return r;
}
FILE *fopen64(const char *p,const char *mode) {
    typedef FILE *(*Fn)(const char *,const char *);static Fn next;if(!next)next=(Fn)resolve(RTLD_NEXT,"fopen64");
    FILE *r=next(p,mode);file_record(p,r?0:-1);return r;
}
static void json_string(FILE *f,const char *p) {
    fputc('"',f);for(;*p;p++){if(*p=='"'||*p=='\\')fputc('\\',f);if((unsigned char)*p>=32)fputc(*p,f);}fputc('"',f);
}
void sub_probe_dump(const char *path) {
    FILE *f=fopen(path,"w");if(!f)abort();unsigned n=atomic_load(&used);fprintf(f,"[\n");
    for(unsigned i=0;i<n;i++) {
        Row *r=&rows[i];fprintf(f,"{\"scope\":%u,\"kind\":%u,\"begin_ns\":%lu,\"end_ns\":%lu,\"tid\":%lu,\"result\":%d,\"u\":[",r->scope,r->kind,r->begin,r->end,r->tid,r->result);
        for(unsigned j=0;j<12;j++)fprintf(f,"%s%lu",j?",":"",r->u[j]);
        fprintf(f,"],\"path\":");json_string(f,r->path);fprintf(f,"}%s\n",i+1<n?",":"");
    }
    fprintf(f,"]\n");fclose(f);
}
