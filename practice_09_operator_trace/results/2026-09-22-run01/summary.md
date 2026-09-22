# Real operator trace validation

Request: `cmpl-practice09-profile-0-9b57ff02`

126 input tokens; 2 output tokens; 127 computed tokens. Warmup excluded from profiler.

6049 host cpu_op events; 772 device tasks; 687 kernel CSV rows.

Device task count includes copies/events; it is not the kernel CSV row count.

| Phase | PyTorch operator | Device kernel | Task / stream | Device duration (us) | Kernel start minus Python scope end (us) |
|---|---|---|---|---|---|
| prefill | `atb::_npu_reshape_and_cache` | `ReshapeAndCacheNdKernel` | 1339 / 46 | 10.2605 | 51.913 |
| prefill | `npu::npu_fused_infer_attention_score` | `FusedInferAttentionScore` | 1341 / 46 | 29.401 | 46.967 |
| decode | `atb::_npu_reshape_and_cache` | `ReshapeAndCacheNdKernel` | 1764 / 46 | 1.86 | 14.556 |
| decode | `npu::npu_fused_infer_attention_score` | `FusedInferAttentionScore` | 1765 / 46 | 21.041 | -15.979 |

Positive last column means device execution started after the annotated Python function returned.
Negative means the device started before that host scope ended; it does not establish completion.

Each example follows an async_npu flow ID and a HostToDevice flow ID, and matches the kernel CSV
by task ID, stream ID, exact start timestamp, name and rounded duration.

Open first_layer_trace.json in a compatible trace viewer; operator_links.json contains exact raw evidence.
These instrumented durations are diagnostic observations, not benchmark numbers.
