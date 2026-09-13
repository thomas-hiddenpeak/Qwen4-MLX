import Foundation

/// Executor-confined KV-only cache attachment. It deliberately has no public
/// initializer or use API: equal offsets do not prove equal token prefixes.
/// The cache must bind namespace + token-prefix key before sharing this handle
/// with a request or using it as the ancestor of another attachment.
///
/// A strong context keeps the arena owners alive. The cache separately admits
/// metadataBytes and retains that permit while any cache/request owns this
/// attachment. Derived request states and their pending graphs require the
/// ordinary request/workspace permit until completion; this is not an RSS cap.
final class QwenPagedKVPrefix {
    let modelOwner: UUID
    let context: QwenPagedKVContext
    let offset: Int
    let attentionStates: [Int: GPUPagedKVPool.State]
    let metadataBytes: Int

    /// Conservative logical permit for immutable per-layer page lists/tables
    /// and Swift/native wrappers. Arena bytes have their own context permit.
    /// The allowance includes 256 bytes per logical page plus 64 KiB per layer;
    /// it is not a measurement of host allocator capacity or process memory.
    static func estimatedMetadataBytes(offset: Int, attentionLayerCount: Int) throws -> Int {
        guard (1...131_072).contains(offset), (1...48).contains(attentionLayerCount) else {
            throw GPUError.invalid("Paged prefix metadata estimate requires a bounded offset and layer count")
        }
        let pages = (offset + 31) / 32
        return 4_096 + attentionLayerCount * (65_536 + pages * 256)
    }

    init(modelOwner: UUID, context: QwenPagedKVContext, offset: Int,
         attentionStates: [Int: GPUPagedKVPool.State]) throws {
        self.modelOwner = modelOwner; self.context = context; self.offset = offset
        self.attentionStates = attentionStates
        metadataBytes = try Self.estimatedMetadataBytes(offset: offset, attentionLayerCount: attentionStates.count)
        try validate(modelOwner: modelOwner, context: context, layerIndices: context.pools.keys.sorted())
    }

    /// Checks only trusted structural provenance. Token-prefix equivalence is
    /// checked by the cache against its request key before this method is used.
    func validate(modelOwner expectedOwner: UUID, context expectedContext: QwenPagedKVContext,
                  layerIndices: [Int]) throws {
        guard modelOwner == expectedOwner, context.modelOwner == expectedOwner,
              context === expectedContext, offset > 0, offset <= context.maximumTokens,
              Set(attentionStates.keys) == Set(layerIndices), Set(context.pools.keys) == Set(layerIndices) else {
            throw GPUError.invalid("Paged prefix model, context, layer set or offset mismatch")
        }
        for layer in layerIndices {
            guard let state = attentionStates[layer], let pool = context.pools[layer],
                  state.belongs(to: pool), state.logicalTokens == offset,
                  state.pageCount == (offset + 31) / 32 else {
                throw GPUError.invalid("Paged prefix contains foreign or inconsistent layer storage")
            }
        }
    }

    var evaluationTensors: [Tensor] {
        get throws { try attentionStates.keys.sorted().map { try attentionStates[$0]!.ready() } }
    }
}
