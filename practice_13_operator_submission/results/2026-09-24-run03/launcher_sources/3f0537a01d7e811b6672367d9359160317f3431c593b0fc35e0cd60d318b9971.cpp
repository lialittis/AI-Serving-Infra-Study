

#include "precompiled.h"



#define PY_SSIZE_T_CLEAN

#define TENSOR_KIND_INPUT 0
#define TENSOR_KIND_OUTPUT 1
#define TENSOR_KIND_INPUT_OUTPUT 2


extern "C" {
  typedef int (* callback)(unsigned int type, void* data, unsigned int len);
  extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
  extern unsigned long int  MsprofSysCycleTime();
  extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
  static unsigned int __MsprofFlagL0  = 0;
  static unsigned int __MsprofFlagL1  = 0;

  int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
    if ((CtrlData == nullptr) || (DataLen == 0U)) {
      return 1;
    }

    if (CtrlType == 1) {
      MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
      if (handle->type >= 6)  // 6 is not used here
        return 1;
      if (handle->type == 1) {  // init - 0  , start - 1
        __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
        __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
      }
    }
    return 0;
  }
}



typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    aclrtPtrAttributes attributes;
    aclError status = aclrtPointerGetAttributes(ptr_info.dev_ptr, &attributes);

    if (status == ACL_SUCCESS) {
      if (attributes.location.type != ACL_MEM_LOCATION_TYPE_DEVICE && attributes.location.type != 4) {
        Py_DECREF(ret);
        PyErr_Format(PyExc_ValueError,
                     "Pointer argument (at %d) cannot be accessed from Triton (cpu tensor?)", idx);
        ptr_info.valid = false;
        return ptr_info;
      }
    } else {
      Py_DECREF(ret);
      PyErr_Format(PyExc_RuntimeError,
                   "Failed to query pointer attributes at argument %d. "
                   "Error code: %d. This may indicate invalid memory address "
                   "or NPU device error.",
                   idx, status);
      ptr_info.valid = false;
      return ptr_info;
      }
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  ptr_info.valid = false;
  return ptr_info;
}


static void _launch(const char* kernelName, const void* func, rtStream_t stream, int gridX, int gridY, int gridZ, std::vector<std::vector<int64_t>> &tensorShapes, std::vector<int> &tensorKinds, int32_t arg0, int32_t arg1, void* arg2, void* arg3, void* arg4, int32_t arg5, int32_t arg6, void* arg7) {
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  std::string name = "";
  name.append(kernelName);
  void *workspace_addr_ptr = NULL;
  uint32_t blockNum4Workspace = gridX * gridY * gridZ;
  
  
  auto launch_call = [=]() -> rtError_t {
    
    uint32_t blockNum = gridX * gridY * gridZ;

    #ifdef ENABLE_GRID_WARN_PRINT
      static bool warned = false;
      if (!warned && blockNum > (uint32_t)48) {
        printf("WARNING: Grid %u > physical limit 48, performance maybe reduced.\n",blockNum);
        warned = true;
    }
    #endif
    
    // set mixBlockNumRation for nodeBasicBlockDim for msprof report
    uint32_t mixBlockNumRation = 0;
    uint32_t nodeBasicBlockDim = (mixBlockNumRation << 16) + blockNum;

    
    rtError_t ret = RT_ERROR_NONE;
    void *ffts_addr = NULL; uint32_t ffts_len; ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    if (ret != RT_ERROR_NONE) return ret;
    // stub argument for workspace
    void *syncBlockLock_ptr = NULL;
    uint16_t ModuleId = 0;
    
    
    struct __attribute__((packed)) {
      void* ffts_addr __attribute__((aligned(8)));
      void* syncBlockLock __attribute__((aligned(8)));
      void* workspace_addr __attribute__((aligned(8)));
      int32_t arg0 __attribute__((aligned(4))); int32_t arg1 __attribute__((aligned(4))); void* arg2 __attribute__((aligned(8))); void* arg3 __attribute__((aligned(8))); void* arg4 __attribute__((aligned(8))); int32_t arg5 __attribute__((aligned(4))); int32_t arg6 __attribute__((aligned(4))); void* arg7 __attribute__((aligned(8)));
      int32_t gridX __attribute__((aligned(4))); int32_t gridY __attribute__((aligned(4))); int32_t gridZ __attribute__((aligned(4)));
      
    } args = {
      static_cast<void*>(ffts_addr),
      nullptr,
      nullptr,
      static_cast<int32_t>(arg0), static_cast<int32_t>(arg1), static_cast<void*>(arg2), static_cast<void*>(arg3), static_cast<void*>(arg4), static_cast<int32_t>(arg5), static_cast<int32_t>(arg6), static_cast<void*>(arg7),
      static_cast<int32_t>(gridX), static_cast<int32_t>(gridY), static_cast<int32_t>(gridZ)
      
    };
    
    unsigned long int beginTime = 0;
    unsigned long int endTime = 0;
    unsigned long int opNameHashID = 0;
    unsigned int threadId = 0;
    char* _kernelName = const_cast<char*>(name.c_str());
    size_t length = name.length();
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      beginTime = MsprofSysCycleTime();
    }

    
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);

    
    
    
    if (__MsprofFlagL0 || __MsprofFlagL1)
    {
      endTime = MsprofSysCycleTime();
      opNameHashID = MsprofGetHashId(_kernelName, length);
      threadId = (unsigned int)(syscall(SYS_gettid));
      MsprofApi info;
      info.level = MSPROF_REPORT_NODE_LEVEL;
      info.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      info.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
      info.threadId = threadId;
      info.reserve = 0;
      info.beginTime = beginTime;
      info.endTime = endTime;
      info.itemId = opNameHashID;
      MsprofReportApi(false, &info);
    }
    if (__MsprofFlagL1)
    {
      MsprofCompactInfo nodeBasicInfo;
      nodeBasicInfo.level = MSPROF_REPORT_NODE_LEVEL;
      nodeBasicInfo.magicNumber = 0x5a5a;      //MSPROF_REPORT_DATA_MAGIC_NUM
      nodeBasicInfo.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
      nodeBasicInfo.threadId = threadId;
      nodeBasicInfo.timeStamp = endTime;
      nodeBasicInfo.data.nodeBasicInfo.opName = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.opType = opNameHashID;
      nodeBasicInfo.data.nodeBasicInfo.taskType = MSPROF_GE_TASK_TYPE_AIV;
      nodeBasicInfo.data.nodeBasicInfo.blockDim = nodeBasicBlockDim;
      MsprofReportCompactInfo(0, static_cast<void *>(&nodeBasicInfo), sizeof(MsprofCompactInfo));

      // 'mix' kernel need to report the ctxID
      if (false > 0) {
        MsprofAdditionalInfo info;
        info.level = MSPROF_REPORT_NODE_LEVEL;
        info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
        info.threadId = threadId;
        info.timeStamp = endTime;
        MsprofContextIdInfo ctxId;
        ctxId.opName = opNameHashID;
        ctxId.ctxIdNum = 1;
        for (uint32_t i = 0; i < ctxId.ctxIdNum; i++) {
          ctxId.ctxIds[i] = i;
        }
        size_t copyLen = sizeof(MsprofContextIdInfo);
        if (copyLen > MSPROF_ADDTIONAL_INFO_DATA_LENGTH) {
          copyLen = MSPROF_ADDTIONAL_INFO_DATA_LENGTH;
        }
        memcpy(info.data, &ctxId, copyLen);
        MsprofReportAdditionalInfo(false, static_cast<void *>(&info), sizeof(MsprofAdditionalInfo));
      }

      // Report tensor info
      int max_tensors_num = tensorShapes.size() < MSPROF_GE_TENSOR_DATA_NUM ? tensorShapes.size() : MSPROF_GE_TENSOR_DATA_NUM;
      MsprofAdditionalInfo tensorInfo;
      tensorInfo.level = MSPROF_REPORT_NODE_LEVEL;
      tensorInfo.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
      tensorInfo.threadId = threadId;
      tensorInfo.timeStamp = endTime;
      auto profTensorData = reinterpret_cast<MsprofTensorInfo *>(tensorInfo.data);
      profTensorData->opName = opNameHashID;
      int tensorCount = 0;
      int dataTypes[MSPROF_GE_TENSOR_DATA_NUM];
      if (tensorShapes.size() > 0) {
        dataTypes[2] = 3;
dataTypes[3] = 9;
dataTypes[4] = 3;
      }
      for (int i = 0; i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM; i++) {
        auto fillTensorData = [&](int index, int tensorType) {
          profTensorData->tensorData[index].tensorType = tensorType;
          profTensorData->tensorData[index].format = 2; // GeDataFormat: ND = 2
          profTensorData->tensorData[index].dataType = dataTypes[i];
          int nDim = tensorShapes[i].size();
          nDim = nDim < MSPROF_GE_TENSOR_DATA_SHAPE_LEN ? nDim : MSPROF_GE_TENSOR_DATA_SHAPE_LEN;
          for (int j = 0; j < nDim; j++) {
            profTensorData->tensorData[index].shape[j] = tensorShapes[i][j];
          }
          for (int j = nDim; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; j++) {
            profTensorData->tensorData[index].shape[j] = 0;
          }
        };
        int tensorType = (i < tensorKinds.size()) ? tensorKinds[i] : 0;  // DeFault tensor type is input
        if (tensorType == TENSOR_KIND_INPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) {
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_INPUT);
          tensorCount++;
        }
        if ((tensorType == TENSOR_KIND_OUTPUT || tensorType == TENSOR_KIND_INPUT_OUTPUT) && tensorCount < MSPROF_GE_TENSOR_DATA_NUM){
          fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);
          tensorCount++;
        }
      }
      profTensorData->tensorNum = tensorCount;
      MsprofReportAdditionalInfo(false, static_cast<void *>(&tensorInfo), sizeof(MsprofAdditionalInfo));
    }

    return ret;
   };
   at_npu::native::OpCommand cmd;
    cmd.Name(name.c_str()).SetCustomHandler(launch_call).Run();
  return;
}

// Extract tensor shape from PyObject
static std::vector<int64_t> _get_tensor_shape(PyObject *tensor) {
  std::vector<int64_t> shape;

  // Early return if tensor is None or null
  if (!tensor || tensor == Py_None) {
    return shape;
  }

  // Calling tensor.size()
  PyObject* size_result = PyObject_CallMethod(tensor, "size", NULL);
  if (!size_result) {
    return shape;
  }
  // Using PySequence_Fast to improve access efficiency
  PyObject* seq = PySequence_Fast(size_result, "Expected a sequence from tensor.size()");
  if (seq) {
    Py_ssize_t len = PySequence_Fast_GET_SIZE(seq);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* dim = items[i];
      if (PyLong_Check(dim)) {
        shape.push_back(PyLong_AsLong(dim));
      }
    }
  }
  Py_DECREF(seq);
  Py_DECREF(size_result);
  return shape;
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX, gridY, gridZ;
  rtStream_t stream;
  const void *function;
  PyObject *packedMetadata = NULL;
  PyObject *launch_metadata = NULL;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  std::vector<std::vector<int64_t>> tensorShapes;
  int32_t _arg0;  int32_t _arg1;  PyObject* _arg2;  PyObject* _arg3;  PyObject* _arg4;  int32_t _arg5;  int32_t _arg6;  PyObject* _arg7; 
  if(!PyArg_ParseTuple(
      args, "iiiKKOOOOiiOOOiiO",
      &gridX, &gridY, &gridZ, &stream, &function,
      &packedMetadata, &launch_metadata,
      &launch_enter_hook, &launch_exit_hook
      , &_arg0, &_arg1, &_arg2, &_arg3, &_arg4, &_arg5, &_arg6, &_arg7
      )
    ) {
    return NULL;
  }
  if (__MsprofFlagL1)
  {
    { auto tmp = _get_tensor_shape(_arg2); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg3); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg4); if (!tmp.empty()) tensorShapes.push_back(tmp); }
{ auto tmp = _get_tensor_shape(_arg7); if (!tmp.empty()) tensorShapes.push_back(tmp); }
  }

  if (launch_enter_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_enter_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }

  // get kernel_name
  PyObject *kernelNameObj = PyDict_GetItemString(packedMetadata, "kernel_name");
  const char *kernelName = PyUnicode_AsUTF8(kernelNameObj);
  // get tensor_kinds
  std::vector<int> tensorKinds;
  PyObject *tensorKindList = PyDict_GetItemString(packedMetadata, "tensor_kinds");
  if (tensorKindList) {
    int size = PyObject_Size(tensorKindList);
    for (int i = 0; i < size; i++) {
      PyObject *kind = PySequence_GetItem(tensorKindList, i);
      tensorKinds.push_back(PyLong_AsLong(kind));
    }
  }

  // raise exception asap
  ; ; DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL;; DevicePtrInfo ptr_info3 = getPointer(_arg3, 3); if (!ptr_info3.valid) return NULL;; DevicePtrInfo ptr_info4 = getPointer(_arg4, 4); if (!ptr_info4.valid) return NULL;; ; ; DevicePtrInfo ptr_info7 = getPointer(_arg7, 7); if (!ptr_info7.valid) return NULL;;
  _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, _arg0, _arg1, ptr_info2.dev_ptr, ptr_info3.dev_ptr, ptr_info4.dev_ptr, _arg5, _arg6, ptr_info7.dev_ptr);
  if (PyErr_Occurred()) {
    return NULL;
  }
  if(launch_exit_hook != Py_None){
    PyObject* args = Py_BuildValue("(O)", launch_metadata);
    PyObject* ret = PyObject_CallObject(launch_exit_hook, args);
    Py_DECREF(args);
    if (!ret)
      return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__triton_launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___triton_launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  
  MsprofRegisterCallback(8, ProfCtrlHandle);      // 8 - CCE defined in msprof headerfile slog.h

  return m;
}
