// Adapted from garnermccloud/mlx-serve src/transformer.zig, fixed commit
// 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1, compileQwen4Hc and hc{Silu,Mix,Inj,Write}Callback.
// Dense HC tail adaptation via the same public MLX compile API.
//
// MIT License — Copyright (c) 2026 David Dalcu
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import CMLX
import Foundation

/// The four compiled tails actually used by the pinned author on dense BF16
/// hyper connections. Share ONE instance across a model's HC modules on its
/// inference thread, so their compiled shape caches are reused.
///
/// This follows the author's compile graph, including whatever intermediate
/// BF16 rounding and multiply/add fusion the same MLX compiler selects. A
/// manually assumed single-FMA write is not substituted for that graph.
/// No state, weights, GPU evaluation or benchmark runs during initialization.
public final class GPUHyperConnectionFused {
    private let siluOperation: HCCompiledOperation
    private let mixOperation: HCCompiledOperation
    private let injectionOperation: HCCompiledOperation
    private let writeOperation: HCCompiledOperation

    public init() throws {
        siluOperation = try HCCompiledOperation(shapeless: true) { result,input in
            hcCompiledResult(result) { try MX.silu(hcCompiledInput(input,0)) }
        }
        // Mean changes its reduction geometry with sequence length, matching
        // compileQwen4Hc's shapeless=false for this callback only.
        mixOperation = try HCCompiledOperation(shapeless: false) { result,input in
            hcCompiledResult(result) {
                let up = try hcCompiledInput(input,0), normalized = try hcCompiledInput(input,1)
                return try MX.mean(MX.mul(MX.sigmoid(up),normalized),axis: 2)
            }
        }
        injectionOperation = try HCCompiledOperation(shapeless: true) { result,input in
            hcCompiledResult(result) {
                let raw = try hcCompiledInput(input,0)
                return try MX.mul(MX.sigmoid(raw),MX.scalar(2,raw.dtype))
            }
        }
        writeOperation = try HCCompiledOperation(shapeless: true) { result,input in
            hcCompiledResult(result) {
                let stream = try hcCompiledInput(input,0), output = try hcCompiledInput(input,1)
                let injection = try hcCompiledInput(input,2)
                return try MX.add(stream,MX.mul(output,injection))
            }
        }
    }

    public func silu(_ down: Tensor) throws -> Tensor {
        guard down.dtype == MLX_BFLOAT16, down.count > 0 else {
            throw GPUError.invalid("Compiled HC SiLU requires nonempty BF16")
        }
        return try siluOperation.apply([down])
    }

    /// normalized and upOutput both [B,S,4,2560]; return mixed [B,S,2560].
    public func readMix(normalized: Tensor, upOutput: Tensor) throws -> Tensor {
        guard normalized.shape == upOutput.shape, normalized.shape.count == 4,
              normalized.shape[0] > 0, normalized.shape[1] > 0,
              normalized.shape[2] == 4, normalized.shape[3] == 2560,
              normalized.dtype == MLX_BFLOAT16, upOutput.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Compiled HC mix requires matching BF16 [B,S,4,2560]")
        }
        return try mixOperation.apply([upOutput,normalized])
    }

    /// Input is the already scaled-weight injection projection [B,S,4].
    public func injection(_ logits: Tensor) throws -> Tensor {
        guard logits.shape.count == 3, logits.shape[0] > 0, logits.shape[1] > 0,
              logits.shape[2] == 4, logits.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Compiled HC injection requires BF16 [B,S,4]")
        }
        return try MX.reshape(injectionOperation.apply([logits]),[logits.shape[0],logits.shape[1],4,1])
    }

    public func write(stream: Tensor, output: Tensor, injection: Tensor) throws -> Tensor {
        let shape = stream.shape
        guard shape.count == 3, shape[0] > 0, shape[1] > 0, shape[2] == 10240,
              output.shape == [shape[0],shape[1],2560], injection.shape == [shape[0],shape[1],4,1],
              [stream,output,injection].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUError.invalid("Compiled HC write shape/dtype mismatch")
        }
        let stream4 = try MX.reshape(stream,[shape[0],shape[1],4,2560])
        let output4 = try MX.reshape(output,[shape[0],shape[1],1,2560])
        return try MX.reshape(writeOperation.apply([stream4,output4,injection]),shape)
    }
}

private typealias HCCallback = @convention(c) (UnsafeMutablePointer<mlx_vector_array>?, mlx_vector_array) -> Int32

/// RAII ensures partial outer-initializer failure frees earlier C closures too.
private final class HCCompiledOperation {
    private let closure: mlx_closure
    init(shapeless: Bool, callback: HCCallback) throws {
        let raw = mlx_closure_new_func(callback)
        defer { _ = mlx_closure_free(raw) }
        guard raw.ctx != nil else { throw GPUError.invalid("Unable to construct HC callback") }
        var compiled = mlx_closure(ctx: nil)
        do {
            try MX.check(mlx_compile(&compiled,raw,shapeless),"compile HC callback")
            guard compiled.ctx != nil else { throw GPUError.invalid("Empty compiled HC callback") }
            closure = compiled
        } catch {
            if compiled.ctx != nil { _ = mlx_closure_free(compiled) }
            throw error
        }
    }
    deinit { _ = mlx_closure_free(closure) }
    func apply(_ tensors: [Tensor]) throws -> Tensor {
        let input = mlx_vector_array_new_data(tensors.map(\.handle),tensors.count)
        var output = mlx_vector_array_new()
        defer { _ = mlx_vector_array_free(input); _ = mlx_vector_array_free(output) }
        try MX.check(mlx_closure_apply(&output,closure,input),"apply compiled HC")
        guard mlx_vector_array_size(output) == 1 else { throw GPUError.invalid("Compiled HC output count mismatch") }
        return try MX.output("compiled HC output") { mlx_vector_array_get(&$0,output,0) }
    }
}

private func hcCompiledInput(_ input: mlx_vector_array, _ index: Int) throws -> Tensor {
    try MX.output("HC callback input") { mlx_vector_array_get(&$0,input,index) }
}

private func hcCompiledResult(_ result: UnsafeMutablePointer<mlx_vector_array>?, _ body: () throws -> Tensor) -> Int32 {
    guard let result else { return -1 }
    do {
        let tensor = try body()
        // Set the vector supplied by the C trampoline, preserving its ownership.
        return mlx_vector_array_set_data(result,[tensor.handle],1)
    } catch {
        // C callbacks cannot propagate Swift errors; MLX's outer checked call
        // turns this status into a throwing API error. No fallback is hidden.
        return -1
    }
}
