#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif
int32_t anemlx_paged_sdpa_version(void);
const char* anemlx_paged_sdpa_last_error(void);
// One immutable identity int32 table per handle, bounded to 1...131072 tokens.
// Caller owns the context; constructed output graphs retain their own inputs.
int32_t anemlx_paged_sdpa_create(void** context, int32_t maximum_tokens);
void anemlx_paged_sdpa_free(void* context);
uint64_t anemlx_paged_sdpa_metadata_bytes(const void* context);
// DSO-wide successful full reader encodings, including any other contexts or
// the generic page-major reader. No reset; not graph calls or GPU completions.
uint64_t anemlx_paged_sdpa_encoded_reads(void);
// Lazy head-major read. Null mask.ctx means no mask; existing K/V are borrowed
// during construction and retained as graph dependencies, never packed/gathered.
int32_t anemlx_paged_sdpa_read(mlx_array* output, const void* context,
    mlx_array queries, mlx_array keys, mlx_array values, mlx_array mask, mlx_stream stream);
// Three scalars: two_pass, blocks, logical_scratch_bytes. Excludes 12288-byte
// output, allocator rounding/cache, metadata, and any concurrent live graphs.
int32_t anemlx_paged_sdpa_dispatch_info(int32_t tokens, mlx_stream stream,
    uint64_t* scalars, size_t count);
#ifdef __cplusplus
}
#endif
