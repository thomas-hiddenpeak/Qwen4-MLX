#pragma once
#include <mlx/c/array.h>
#include <mlx/c/stream.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t anemlx_paged_kv_pool_version(void);
const char* anemlx_paged_kv_pool_last_error(void);
int32_t anemlx_paged_kv_pool_create(void** pool, int32_t physical_pages, mlx_stream stream);
void anemlx_paged_kv_pool_free(void* pool);
int32_t anemlx_paged_kv_pool_import(void** state, const void* pool, mlx_array keys, mlx_array values);
int32_t anemlx_paged_kv_pool_fork(void** state, const void* source);
int32_t anemlx_paged_kv_pool_append(void** state, const void* source, mlx_array keys, mlx_array values);
void anemlx_paged_kv_pool_state_free(void* state);
int32_t anemlx_paged_kv_pool_state_info(const void* state, uint64_t* scalars, size_t count);
int32_t anemlx_paged_kv_pool_page_ids(const void* state, int32_t* pages, size_t count);
int32_t anemlx_paged_kv_pool_ready(mlx_array* result, const void* state);
int32_t anemlx_paged_kv_pool_read(mlx_array* result, const void* state, mlx_array query, mlx_array mask);
int32_t anemlx_paged_kv_pool_materialize(mlx_array* keys, mlx_array* values, const void* state);
int32_t anemlx_paged_kv_pool_statistics(const void* pool, uint64_t* scalars, size_t count);

#ifdef __cplusplus
}
#endif
