#!/usr/bin/env python3
"""Optional rank-one external affine-Q4 banks, preserving original byte storage.

The selected GEMV and grouped prefill kernels use explicit row-major addressing.
Geometry is specialized per kernel; no graph reshape recreates rank-three input
tensors. CPU callbacks reshape only to reuse the established authoring oracles.
"""
from __future__ import annotations

from functools import cache
import torch

from coreai_q4_metal import Q4_METAL_SOURCE, selected_q4_reference
from coreai_q4_grouped import GEMM_SOURCE, grouped_reference
from coreai_q4_gateup import GATEUP_SOURCE, gateup_reference


def _geometry(experts, outputs, inputs):
    if min(experts, outputs, inputs) < 1 or inputs % 64:
        raise ValueError('Positive affine-Q4 geometry with group64 required')
    return experts, outputs, inputs


def _tile(block, columns, inner):
    if block not in (16, 32) or columns not in (32, 64) or inner not in (64, 128):
        raise ValueError('Unsupported flat grouped Q4 tile')


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Q4 source changed; expected exactly one flat addressing target: {old}')
    return source.replace(old, new)


def _reshape(packed, scales, biases, experts, outputs, inputs):
    return (packed.reshape(experts, outputs, inputs // 4),
            scales.reshape(experts, outputs, inputs // 64),
            biases.reshape(experts, outputs, inputs // 64))


def _validate(x, packed, scales, biases, experts, outputs, inputs):
    if (x.dtype != torch.float16 or x.shape[-1] != inputs or packed.dtype != torch.int16
            or scales.dtype != torch.float16 or biases.dtype != torch.float16
            or packed.shape != (experts * outputs * (inputs // 4),)
            or scales.shape != (experts * outputs * (inputs // 64),)
            or biases.shape != scales.shape):
        raise ValueError('Expected original contiguous rank-one I16/FP16 affine-Q4 bank')


@cache
def get_flat_q4_kernel(experts, outputs, inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    _geometry(experts, outputs, inputs)

    def reference(x: torch.Tensor, ids: torch.Tensor, packed: torch.Tensor,
                  scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        return selected_q4_reference(x, ids, *_reshape(packed, scales, biases, experts, outputs, inputs))

    source = _replace_once(Q4_METAL_SOURCE, 'packed.get_extent(1)', str(outputs))
    source = _replace_once(source, 'packed.get_extent(0)', str(inputs // 4))
    source = _replace_once(source, 'packed.get_extent(2)', str(experts))
    source = _replace_once(source, 'packed[word, row, expert]', 'packed[(uint(expert)*out_size+row)*words+word]')
    source = _replace_once(source, 'scales[affine_group, row, expert]',
                            f'scales[(uint(expert)*out_size+row)*{inputs // 64}u+affine_group]')
    source = _replace_once(source, 'biases[affine_group, row, expert]',
                            f'biases[(uint(expert)*out_size+row)*{inputs // 64}u+affine_group]')
    return TorchMetalKernel(f'qwen_affine_q4_flat_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x', 'ids', 'packed', 'scales', 'biases'], result_names=['output'],
        src=source, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('lane', 'uint', 'thread_index_in_simdgroup'),
                      MetalParameter('simd', 'uint', 'simdgroup_index_in_threadgroup')])


def _grouped_source(source, experts, outputs, inputs, block, columns, inner, prefixes):
    _geometry(experts, outputs, inputs)
    _tile(block, columns, inner)
    if prefixes == ('',):
        target = 'const int K=int(x.get_extent(0)), N=int(packed.get_extent(1));'
    elif prefixes == ('gate_', 'up_'):
        target = 'const int K=int(x.get_extent(0)),N=int(gate_packed.get_extent(1));'
    else:
        raise ValueError('Unknown Q4 weight prefix layout')
    source = _replace_once(source, target, f'const int K={inputs}, N={outputs};')
    for prefix in prefixes:
        source = _replace_once(source, f'{prefix}packed[k/4,n,expert]',
                                f'{prefix}packed[(expert*N+n)*(K/4)+k/4]')
        for affine in ('scales', 'biases'):
            source = _replace_once(source, f'{prefix}{affine}[k/64,n,expert]',
                                    f'{prefix}{affine}[(expert*N+n)*(K/64)+k/64]')
    return source.replace('BM', str(block)).replace('BN', str(columns)).replace('BK', str(inner))


@cache
def get_flat_grouped_kernel(experts, outputs, inputs, block=16, columns=32, inner=64):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, packed: torch.Tensor,
                  scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        return grouped_reference(x, plan, *_reshape(packed, scales, biases, experts, outputs, inputs))

    source = _grouped_source(GEMM_SOURCE, experts, outputs, inputs, block, columns, inner, ('',))
    return TorchMetalKernel(f'qwen_q4_grouped_flat_e{experts}_n{outputs}_k{inputs}_m{block}_n{columns}_k{inner}_v1',
        input_names=['x', 'plan', 'packed', 'scales', 'biases'], result_names=['output'],
        src=source, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


@cache
def get_flat_gateup_kernel(experts, outputs, inputs, block=16, columns=32, inner=64):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                  up_packed: torch.Tensor, up_scales: torch.Tensor,
                  up_biases: torch.Tensor) -> torch.Tensor:
        return gateup_reference(x, plan,
            *_reshape(gate_packed, gate_scales, gate_biases, experts, outputs, inputs),
            *_reshape(up_packed, up_scales, up_biases, experts, outputs, inputs))

    source = _grouped_source(GATEUP_SOURCE, experts, outputs, inputs, block, columns, inner, ('gate_', 'up_'))
    return TorchMetalKernel(f'qwen_q4_gateup_flat_e{experts}_n{outputs}_k{inputs}_m{block}_n{columns}_k{inner}_v1',
        input_names=['x', 'plan', 'gate_packed', 'gate_scales', 'gate_biases', 'up_packed', 'up_scales', 'up_biases'],
        result_names=['output'], src=source, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class FlatMetalPackedQ4(torch.nn.Module):
    """Same buffer names/order and backing storage as the original projection."""

    def __init__(self, original):
        super().__init__()
        if original.group_size != 64 or original.packed.ndim != 3:
            raise ValueError('Expected an original rank-three group64 projection')
        self.expert_count, self.output_size, words = original.packed.shape
        self.input_size, self.group_size = words * 4, 64
        _geometry(self.expert_count, self.output_size, self.input_size)
        for name in ('packed', 'scales', 'biases'):
            value = getattr(original, name)
            if not value.is_contiguous():
                raise ValueError('Flattening must preserve contiguous original storage')
            self.register_buffer(name, value.view(-1))
        _validate(torch.empty(0, self.input_size, dtype=torch.float16), self.packed,
                  self.scales, self.biases, self.expert_count, self.output_size, self.input_size)

    @property
    def geometry(self):
        return self.expert_count, self.output_size, self.input_size

    def forward(self, x, ids):
        _validate(x, self.packed, self.scales, self.biases, *self.geometry)
        if x.ndim != 3 or x.shape[1] != 1 or ids.shape != (x.shape[0],) or ids.dtype != torch.int32:
            raise ValueError('Expected FP16 selected x[B,1,K] and I32 ids[B]')
        return get_flat_q4_kernel(*self.geometry)(x, ids, self.packed, self.scales, self.biases,
            threads_per_grid=(((self.output_size + 3) // 4) * 128, x.shape[0], 1),
            threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], 1, self.output_size]])


def flat_grouped_linear(x, plan, projection, block, columns, inner):
    _validate(x, projection.packed, projection.scales, projection.biases, *projection.geometry)
    return get_flat_grouped_kernel(*projection.geometry, block, columns, inner)(
        x, plan, projection.packed, projection.scales, projection.biases,
        threads_per_grid=(((projection.output_size + columns - 1) // columns) * 128, plan.shape[0] - 1, 1),
        threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], projection.output_size]])


def flat_grouped_gateup(x, plan, gate, up, block, columns, inner):
    if gate.geometry != up.geometry:
        raise ValueError('Gate/up geometries must match')
    for projection in (gate, up):
        _validate(x, projection.packed, projection.scales, projection.biases, *projection.geometry)
    return get_flat_gateup_kernel(*gate.geometry, block, columns, inner)(x, plan,
        gate.packed, gate.scales, gate.biases, up.packed, up.scales, up.biases,
        threads_per_grid=(((gate.output_size + columns - 1) // columns) * 128, plan.shape[0] - 1, 1),
        threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], gate.output_size]])


def flat_moe_kernels(moe):
    from export_coreai_q4_moe import PROJECTIONS
    kernels = []
    for name in PROJECTIONS:
        projection = getattr(moe.decode, name)
        kernels += [get_flat_q4_kernel(*projection.geometry),
                    get_flat_grouped_kernel(*projection.geometry, moe.block, moe.columns, moe.inner)]
    if moe.fuse_gateup:
        kernels.append(get_flat_gateup_kernel(*moe.decode.gate_proj.geometry, moe.block, moe.columns, moe.inner))
    return list(dict.fromkeys(kernels))


def flatten_moe_weights(module):
    """Mutate only Q4 views/dispatch; return kernels to add before conversion.

    Call before externalizable_buffers. Existing per-layer weight files remain
    valid: buffer names/order, dtype, byte lengths and byte contents are invariant.
    Repeated installation is harmless and returns the same kernel registrations.
    """
    from coreai_moe_chunk import ChunkQ4MoE
    from export_coreai_q4_moe import PROJECTIONS
    before = [(name, value.dtype, value.numel(), value.data_ptr()) for name, value in module.named_buffers()]
    chunks = [child for child in module.modules() if isinstance(child, ChunkQ4MoE)]
    if not chunks:
        raise ValueError('No ChunkQ4MoE found to flatten')
    kernels = []
    for moe in chunks:
        for name in PROJECTIONS:
            old = getattr(moe.decode, name)
            if not isinstance(old, FlatMetalPackedQ4):
                setattr(moe.decode, name, FlatMetalPackedQ4(old))
        moe.flat_weights = True
        kernels += flat_moe_kernels(moe)
    after = [(name, value.dtype, value.numel(), value.data_ptr()) for name, value in module.named_buffers()]
    if before != after:
        raise AssertionError('Flattening changed buffer order/name/dtype/size/storage')
    return list(dict.fromkeys(kernels))
