// Fixed BF16 K/V page movement. Copy words exactly; no arithmetic conversion.
#include <metal_stdlib>
using namespace metal;

kernel void anemlx_kv_pool_copy_tail(
    device ushort* keys [[buffer(0)]], device ushort* values [[buffer(1)]],
    constant int& source_page [[buffer(2)]], constant int& destination_page [[buffer(3)]],
    constant int& rows [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(2 * rows * 256)) return;
  const uint head = i / uint(rows * 256);
  const uint row_dim = i % uint(rows * 256);
  const uint source = uint(source_page * 16384) + head * 8192 + row_dim;
  const uint destination = uint(destination_page * 16384) + head * 8192 + row_dim;
  keys[destination] = keys[source]; values[destination] = values[source];
}

kernel void anemlx_kv_pool_write_rows(
    const device ushort* input_k [[buffer(0)]], const device ushort* input_v [[buffer(1)]],
    device ushort* arena_k [[buffer(2)]], device ushort* arena_v [[buffer(3)]],
    const device int* destination_pages [[buffer(4)]],
    constant int& old_tokens [[buffer(5)]], constant int& rows [[buffer(6)]],
    constant ulong& k_head_stride [[buffer(7)]], constant ulong& k_row_stride [[buffer(8)]],
    constant ulong& v_head_stride [[buffer(9)]], constant ulong& v_row_stride [[buffer(10)]],
    device uint* ticket [[buffer(11)]], uint i [[thread_position_in_grid]]) {
  if (i >= uint(2 * rows * 256)) return;
  const uint head = i / uint(rows * 256), rd = i % uint(rows * 256);
  const uint row = rd / 256, dim = rd % 256;
  const uint logical = uint(old_tokens) + row;
  const uint dest = uint(destination_pages[logical / 32]) * 16384 +
                    head * 8192 + (logical % 32) * 256 + dim;
  arena_k[dest] = input_k[head * k_head_stride + row * k_row_stride + dim];
  arena_v[dest] = input_v[head * v_head_stride + row * v_row_stride + dim];
  if (i == 0) ticket[0] = uint(old_tokens + rows);
}

kernel void anemlx_kv_pool_materialize(
    const device ushort* arena_k [[buffer(0)]], const device ushort* arena_v [[buffer(1)]],
    const device int* pages [[buffer(2)]], device ushort* output_k [[buffer(3)]],
    device ushort* output_v [[buffer(4)]], constant int& tokens [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= uint(2 * tokens * 256)) return;
  const uint head = i / uint(tokens * 256), rd = i % uint(tokens * 256);
  const uint row = rd / 256, dim = rd % 256;
  const uint source = uint(pages[row / 32]) * 16384 + head * 8192 + (row % 32) * 256 + dim;
  output_k[i] = arena_k[source]; output_v[i] = arena_v[source];
}
