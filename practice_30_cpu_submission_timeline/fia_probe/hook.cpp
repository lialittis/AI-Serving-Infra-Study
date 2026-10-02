// Diagnostic-only forwarding hook. Arguments/return values are preserved.
#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v3.h>
#include <openssl/sha.h>
#include <atomic>
#include <errno.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct Row { unsigned scope; uint64_t begin,end,tid,u[24]; int result; double scalar; char kind[80],text[1024]; };
static Row rows[16384];
static std::atomic<unsigned> used{0},scope{0};
static void *resolve(void *handle,const char *name) {
 static auto fn=(void*(*)(void*,const char*))dlvsym(RTLD_NEXT,"dlsym","GLIBC_2.2.5");
 if(!fn)abort();return fn(handle,name);
}
static uint64_t now(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
static Row begin(const char *kind){Row r{};r.scope=scope.load();r.begin=now();r.tid=syscall(SYS_gettid);snprintf(r.kind,sizeof(r.kind),"%s",kind);return r;}
static void record(Row r){if(!r.scope)return;if(!r.end)r.end=now();unsigned i=used.fetch_add(1);if(i>=16384)abort();rows[i]=r;}
extern "C" void fia_scope(unsigned n){scope.store(n);}

#include "generated.inc"

static void file_record(const char *p,int result){
 if(!p || !strstr(p,"/Ascend/") || !strcasestr(p,"fused_infer_attention"))return;
 Row r=begin("file_open");r.result=result;snprintf(r.text,sizeof(r.text),"%s",p);record(r);
}
extern "C" int open(const char *p,int flags,...){
 static auto fn=(int(*)(const char*,int,...))resolve(RTLD_NEXT,"open");
 mode_t m=0;if(flags&O_CREAT){va_list a;va_start(a,flags);m=va_arg(a,int);va_end(a);}
 int r=fn(p,flags,m);int saved=errno;file_record(p,r);errno=saved;return r;
}
extern "C" int open64(const char *p,int flags,...){
 static auto fn=(int(*)(const char*,int,...))resolve(RTLD_NEXT,"open64");
 mode_t m=0;if(flags&O_CREAT){va_list a;va_start(a,flags);m=va_arg(a,int);va_end(a);}
 int r=fn(p,flags,m);int saved=errno;file_record(p,r);errno=saved;return r;
}
extern "C" FILE *fopen(const char *p,const char *mode){
 static auto fn=(FILE*(*)(const char*,const char*))resolve(RTLD_NEXT,"fopen");
 FILE *r=fn(p,mode);int saved=errno;file_record(p,r?0:-1);errno=saved;return r;
}
extern "C" FILE *fopen64(const char *p,const char *mode){
 static auto fn=(FILE*(*)(const char*,const char*))resolve(RTLD_NEXT,"fopen64");
 FILE *r=fn(p,mode);int saved=errno;file_record(p,r?0:-1);errno=saved;return r;
}
static void quoted(FILE *f,const char *s){fputc('"',f);for(;*s;s++){if(*s=='"'||*s=='\\')fputc('\\',f);if((unsigned char)*s>=32)fputc(*s,f);}fputc('"',f);}
extern "C" void fia_dump(const char *path){
 FILE *f=fopen(path,"w");if(!f)abort();unsigned n=used.load();fprintf(f,"[\n");
 for(unsigned i=0;i<n;i++){const Row &r=rows[i];
  fprintf(f,"{\"scope\":%u,\"begin_ns\":%lu,\"end_ns\":%lu,\"tid\":%lu,\"result\":%d,\"scalar\":%.17g,\"kind\":",r.scope,r.begin,r.end,r.tid,r.result,r.scalar);
  quoted(f,r.kind);fprintf(f,",\"text\":");quoted(f,r.text);fprintf(f,",\"u\":[");
  for(int j=0;j<24;j++)fprintf(f,"%s%lu",j?",":"",r.u[j]);fprintf(f,"]}%s\n",i+1<n?",":"");
 }fprintf(f,"]\n");fclose(f);
}
