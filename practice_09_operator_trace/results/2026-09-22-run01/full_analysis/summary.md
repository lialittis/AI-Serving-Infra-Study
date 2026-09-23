# Complete operator audit

Only the recorded Qwen2.5 workload is covered.

- Host events: 6049 (106 distinct names, including diagnostic labels).
- Device tasks: 772 (33 distinct names).
- Both torch and CANN flow IDs resolved: 770.
- Kernel CSV rows verified: 687.
- Enqueue/dequeue pairs verified: 748.
- Unattributed: PROFILING_ENABLE, PROFILING_DISABLE (profiler control, not model kernels).

Route counts partition device tasks; wrapper/host events are separate dimensions.

| Route | Device tasks |
|---|---:|
| ascend_custom_cpp | 96 |
| atb | 72 |
| aten_to_aclnn | 342 |
| profiler_control | 2 |
| runtime_copy | 79 |
| runtime_event | 4 |
| torch_npu | 122 |
| triton | 55 |

| Device kernel/task name | Count | Route |
|---|---:|---|
| `AddRmsNormBias` | 96 | ascend_custom_cpp |
| `EVENT_RECORD` | 4 | runtime_event |
| `FusedInferAttentionScore` | 48 | torch_npu |
| `MEMCPY_ASYNC` | 79 | runtime_copy |
| `PROFILING_DISABLE` | 1 | profiler_control |
| `PROFILING_ENABLE` | 1 | profiler_control |
| `ReshapeAndCacheNdKernel` | 48 | atb |
| `RmsNorm` | 2 | torch_npu |
| `SwiGlu` | 48 | torch_npu |
| `_compute_slot_mapping_kernel` | 2 | triton |
| `_triton_rope` | 48 | triton |
| `aclnnAdd_AddAiCore_Add` | 4 | aten_to_aclnn |
| `aclnnAddmm_CastAiCore_Cast` | 48 | aten_to_aclnn |
| `aclnnAddmm_MatMulCommon_MatMulV2` | 48 | aten_to_aclnn |
| `aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2` | 2 | aten_to_aclnn |
| `aclnnArgMax_CastAiCore_Cast` | 2 | aten_to_aclnn |
| `aclnnContiguous_SliceAiCore_Slice` | 24 | torch_npu |
| `aclnnEmbedding_GatherV2AiCore_GatherV2` | 2 | aten_to_aclnn |
| `aclnnEqScalar_EqualAiCore_Equal` | 1 | aten_to_aclnn |
| `aclnnGtScalar_CastAiCore_Cast` | 4 | aten_to_aclnn |
| `aclnnGtScalar_GreaterAiCore_Greater` | 4 | aten_to_aclnn |
| `aclnnIndexSelect_GatherV3AiCore_GatherV3` | 4 | aten_to_aclnn |
| `aclnnIndex_IndexAiCore_Index` | 4 | aten_to_aclnn |
| `aclnnInplaceCopy_CastAiCore_Cast` | 8 | aten_to_aclnn |
| `aclnnInplaceCopy_SliceAiCore_Slice` | 74 | atb, aten_to_aclnn |
| `aclnnInplaceFillScalar_FillAiCore_Fill` | 4 | aten_to_aclnn |
| `aclnnInplaceMaskedFillScalar_MaskedFillAiCpu_MaskedFill` | 1 | aten_to_aclnn |
| `aclnnInplaceZero_ZerosLikeAiCore_ZerosLike` | 4 | aten_to_aclnn |
| `aclnnMatmul_MatMulCommon_MatMulV2` | 146 | aten_to_aclnn |
| `aclnnRepeat_TileAiCore_Tile` | 4 | aten_to_aclnn |
| `aclnnSubs_SubAiCore_Sub` | 2 | aten_to_aclnn |
| `apply_all_penalties_kernel` | 2 | triton |
| `token_bin_counts_and_mask_kernel` | 3 | triton |

CPU events without a correlated device task are not automatically no-ops: they may
change metadata, allocate memory, synchronize, or do host work. Zero device work is
a statement about this trace, not every possible input or implementation.

outside_annotated_model includes sampling and supporting operations. No unobserved Python
scope or layer number is assigned to those events. Nested CPU scopes use same-lane interval
containment; device attribution uses explicit flow IDs, not temporal proximity.
