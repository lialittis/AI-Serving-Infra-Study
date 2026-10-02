static decltype(&aclnnFusedInferAttentionScoreV3GetWorkspaceSize) real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize;
extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional, const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional, const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional, const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional, const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional, const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional, const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional, const aclTensor *queryRopeOptional, 
    const aclTensor *keyRopeOptional, const aclTensor *keyRopeAntiquantScaleOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor) {
 if(!real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize)real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize=(decltype(real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize))resolve(RTLD_NEXT,"aclnnFusedInferAttentionScoreV3GetWorkspaceSize");
 if(!real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize)abort();Row r=begin("aclnnFusedInferAttentionScoreV3GetWorkspaceSize");
 auto result=real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize(query,key,value,pseShiftOptional,attenMaskOptional,actualSeqLengthsOptional,actualSeqLengthsKvOptional,deqScale1Optional,quantScale1Optional,deqScale2Optional,quantScale2Optional,quantOffset2Optional,antiquantScaleOptional,antiquantOffsetOptional,blockTableOptional,queryPaddingSizeOptional,kvPaddingSizeOptional,keyAntiquantScaleOptional,keyAntiquantOffsetOptional,valueAntiquantScaleOptional,valueAntiquantOffsetOptional,keySharedPrefixOptional,valueSharedPrefixOptional,actualSharedPrefixLenOptional,queryRopeOptional,keyRopeOptional,keyRopeAntiquantScaleOptional,numHeads,scaleValue,preTokens,nextTokens,inputLayout,numKeyValueHeads,sparseMode,innerPrecise,blockSize,antiquantMode,softmaxLseFlag,keyAntiquantMode,valueAntiquantMode,attentionOut,softmaxLse,workspaceSize,executor);r.end=now();r.result=result;
 
 r.u[0]=workspaceSize?*workspaceSize:0;r.u[1]=executor?(uintptr_t)*executor:0;
 r.u[2]=numHeads;r.u[3]=numKeyValueHeads;r.u[4]=preTokens;r.u[5]=nextTokens;
 r.u[6]=sparseMode;r.u[7]=innerPrecise;r.u[8]=blockSize;r.u[9]=antiquantMode;
 r.u[10]=softmaxLseFlag;r.u[11]=keyAntiquantMode;r.u[12]=valueAntiquantMode;
 r.scalar=scaleValue;snprintf(r.text,sizeof(r.text),"%s",inputLayout);
 r.u[13]=(uintptr_t)query;r.u[14]=(uintptr_t)key;r.u[15]=(uintptr_t)value;
 r.u[16]=(uintptr_t)attenMaskOptional;r.u[17]=(uintptr_t)blockTableOptional;
 r.u[18]=(uintptr_t)attentionOut;r.u[19]=(uintptr_t)softmaxLse;
 record(r);return result;
}

static decltype(&aclnnFusedInferAttentionScoreV3) real_aclnnFusedInferAttentionScoreV3;
extern "C" aclnnStatus aclnnFusedInferAttentionScoreV3(void *workspace,
                                                                                   uint64_t workspaceSize,
                                                                                   aclOpExecutor *executor,
                                                                                   const aclrtStream stream) {
 if(!real_aclnnFusedInferAttentionScoreV3)real_aclnnFusedInferAttentionScoreV3=(decltype(real_aclnnFusedInferAttentionScoreV3))resolve(RTLD_NEXT,"aclnnFusedInferAttentionScoreV3");
 if(!real_aclnnFusedInferAttentionScoreV3)abort();Row r=begin("aclnnFusedInferAttentionScoreV3");
 auto result=real_aclnnFusedInferAttentionScoreV3(workspace,workspaceSize,executor,stream);r.end=now();r.result=result;
 r.u[0]=(uintptr_t)workspace;r.u[1]=workspaceSize;r.u[2]=(uintptr_t)executor;r.u[3]=(uintptr_t)stream;record(r);return result;
}

static decltype(&aclrtBinaryLoadFromFile) real_aclrtBinaryLoadFromFile;
extern "C" aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions *options,
                                                     aclrtBinHandle *binHandle) {
 if(!real_aclrtBinaryLoadFromFile)real_aclrtBinaryLoadFromFile=(decltype(real_aclrtBinaryLoadFromFile))resolve(RTLD_NEXT,"aclrtBinaryLoadFromFile");
 if(!real_aclrtBinaryLoadFromFile)abort();Row r=begin("aclrtBinaryLoadFromFile");
 auto result=real_aclrtBinaryLoadFromFile(binPath,options,binHandle);r.end=now();r.result=result;
 snprintf(r.text,sizeof(r.text),"%s",binPath);r.u[0]=binHandle?(uintptr_t)*binHandle:0;record(r);return result;
}

static decltype(&aclrtBinaryLoadFromData) real_aclrtBinaryLoadFromData;
extern "C" aclError aclrtBinaryLoadFromData(const void *data, size_t length,
    const aclrtBinaryLoadOptions *options, aclrtBinHandle *binHandle) {
 if(!real_aclrtBinaryLoadFromData)real_aclrtBinaryLoadFromData=(decltype(real_aclrtBinaryLoadFromData))resolve(RTLD_NEXT,"aclrtBinaryLoadFromData");
 if(!real_aclrtBinaryLoadFromData)abort();Row r=begin("aclrtBinaryLoadFromData");if(r.scope && data && length){unsigned char h[32];SHA256((const unsigned char*)data,length,h);for(int i=0;i<32;i++)snprintf(r.text+2*i,3,"%02x",h[i]);}
 auto result=real_aclrtBinaryLoadFromData(data,length,options,binHandle);r.end=now();r.result=result;
 r.u[0]=binHandle?(uintptr_t)*binHandle:0;r.u[1]=length;record(r);return result;
}

static decltype(&aclrtBinaryGetFunction) real_aclrtBinaryGetFunction;
extern "C" aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char *kernelName,
                                                    aclrtFuncHandle *funcHandle) {
 if(!real_aclrtBinaryGetFunction)real_aclrtBinaryGetFunction=(decltype(real_aclrtBinaryGetFunction))resolve(RTLD_NEXT,"aclrtBinaryGetFunction");
 if(!real_aclrtBinaryGetFunction)abort();Row r=begin("aclrtBinaryGetFunction");
 auto result=real_aclrtBinaryGetFunction(binHandle,kernelName,funcHandle);r.end=now();r.result=result;
 r.u[0]=(uintptr_t)binHandle;r.u[1]=funcHandle?(uintptr_t)*funcHandle:0;snprintf(r.text,sizeof(r.text),"%s",kernelName);record(r);return result;
}

static decltype(&aclrtBinaryGetFunctionByEntry) real_aclrtBinaryGetFunctionByEntry;
extern "C" aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry,
                                                           aclrtFuncHandle *funcHandle) {
 if(!real_aclrtBinaryGetFunctionByEntry)real_aclrtBinaryGetFunctionByEntry=(decltype(real_aclrtBinaryGetFunctionByEntry))resolve(RTLD_NEXT,"aclrtBinaryGetFunctionByEntry");
 if(!real_aclrtBinaryGetFunctionByEntry)abort();Row r=begin("aclrtBinaryGetFunctionByEntry");
 auto result=real_aclrtBinaryGetFunctionByEntry(binHandle,funcEntry,funcHandle);r.end=now();r.result=result;
 r.u[0]=(uintptr_t)binHandle;r.u[1]=funcHandle?(uintptr_t)*funcHandle:0;r.u[2]=funcEntry;record(r);return result;
}

static decltype(&aclrtLaunchKernelWithHostArgs) real_aclrtLaunchKernelWithHostArgs;
extern "C" aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle, uint32_t numBlocks,
                                                           aclrtStream stream, aclrtLaunchKernelCfg *cfg,
                                                           void *hostArgs, size_t argsSize,
                                                           aclrtPlaceHolderInfo *placeHolderArray,
                                                           size_t placeHolderNum) {
 if(!real_aclrtLaunchKernelWithHostArgs)real_aclrtLaunchKernelWithHostArgs=(decltype(real_aclrtLaunchKernelWithHostArgs))resolve(RTLD_NEXT,"aclrtLaunchKernelWithHostArgs");
 if(!real_aclrtLaunchKernelWithHostArgs)abort();Row r=begin("aclrtLaunchKernelWithHostArgs");
 auto result=real_aclrtLaunchKernelWithHostArgs(funcHandle,numBlocks,stream,cfg,hostArgs,argsSize,placeHolderArray,placeHolderNum);r.end=now();r.result=result;
 r.u[0]=(uintptr_t)funcHandle;r.u[1]=numBlocks;r.u[2]=(uintptr_t)stream;r.u[3]=argsSize;r.u[4]=placeHolderNum;record(r);return result;
}

extern "C" void *dlsym(void *handle,const char *name) noexcept {
 void *p=resolve(handle,name);
if(!strcmp(name,"aclnnFusedInferAttentionScoreV3GetWorkspaceSize") && p){real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize=(decltype(real_aclnnFusedInferAttentionScoreV3GetWorkspaceSize))p;return (void*)&aclnnFusedInferAttentionScoreV3GetWorkspaceSize;}
if(!strcmp(name,"aclnnFusedInferAttentionScoreV3") && p){real_aclnnFusedInferAttentionScoreV3=(decltype(real_aclnnFusedInferAttentionScoreV3))p;return (void*)&aclnnFusedInferAttentionScoreV3;}
if(!strcmp(name,"aclrtBinaryLoadFromFile") && p){real_aclrtBinaryLoadFromFile=(decltype(real_aclrtBinaryLoadFromFile))p;return (void*)&aclrtBinaryLoadFromFile;}
if(!strcmp(name,"aclrtBinaryLoadFromData") && p){real_aclrtBinaryLoadFromData=(decltype(real_aclrtBinaryLoadFromData))p;return (void*)&aclrtBinaryLoadFromData;}
if(!strcmp(name,"aclrtBinaryGetFunction") && p){real_aclrtBinaryGetFunction=(decltype(real_aclrtBinaryGetFunction))p;return (void*)&aclrtBinaryGetFunction;}
if(!strcmp(name,"aclrtBinaryGetFunctionByEntry") && p){real_aclrtBinaryGetFunctionByEntry=(decltype(real_aclrtBinaryGetFunctionByEntry))p;return (void*)&aclrtBinaryGetFunctionByEntry;}
if(!strcmp(name,"aclrtLaunchKernelWithHostArgs") && p){real_aclrtLaunchKernelWithHostArgs=(decltype(real_aclrtLaunchKernelWithHostArgs))p;return (void*)&aclrtLaunchKernelWithHostArgs;}
 return p;
}
