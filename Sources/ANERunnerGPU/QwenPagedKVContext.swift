import ANERunnerCore
import Foundation

/// Explicit single-executor experiment. One physical arena per full-attention
/// layer can back several immutable model-state branches. The native pools keep
/// their arena reservations until the last lazy graph/GPU completion releases
/// them. Request, recurrent/QSA state and export workspace remain separately
/// admitted; this ledger is not a process-memory limit.
public final class QwenPagedKVContext {
    let modelOwner: UUID
    let pools: [Int: GPUPagedKVPool]
    public let maximumPagesPerLayer: Int
    public var maximumTokens: Int { min(maximumPagesPerLayer * 32, 131_072) }

    init(modelOwner: UUID, pools: [Int: GPUPagedKVPool], maximumPagesPerLayer: Int) {
        self.modelOwner = modelOwner
        self.pools = pools
        self.maximumPagesPerLayer = maximumPagesPerLayer
    }

    /// Each dictionary entry is a unique physical arena, independent of the
    /// number of branches referring to it. Free pages still retain allocation.
    public var layerStatistics: [Int: GPUPagedKVPool.Statistics] {
        get throws { try pools.mapValues { try $0.statistics } }
    }

    public var reservedArenaBytes: Int { pools.values.reduce(0) { $0 + $1.admittedBytes } }
}
