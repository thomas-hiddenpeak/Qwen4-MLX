import ANERunnerCore
import Foundation

typealias QwenPagedDecodePageLease = QwenPagedPageAdmission.Lease

/// Explicit single-executor experiment. One physical arena per full-attention
/// layer can back several immutable model-state branches. The native pools keep
/// their arena reservations until the last lazy graph/GPU completion releases
/// them. Request, recurrent/QSA state and export workspace remain separately
/// admitted; this ledger is not a process-memory limit.
public final class QwenPagedKVContext {
    let modelOwner: UUID
    let pools: [Int: GPUPagedKVPool]
    private let pageAdmission: QwenPagedPageAdmission
    public let maximumPagesPerLayer: Int
    public var maximumTokens: Int { min(maximumPagesPerLayer * 32, 131_072) }

    init(modelOwner: UUID, pools: [Int: GPUPagedKVPool], maximumPagesPerLayer: Int) throws {
        self.modelOwner = modelOwner
        self.pools = pools
        self.maximumPagesPerLayer = maximumPagesPerLayer
        pageAdmission = try QwenPagedPageAdmission(maximumPagesPerLayer: maximumPagesPerLayer)
    }

    /// Each dictionary entry is a unique physical arena, independent of the
    /// number of branches referring to it. Free pages still retain allocation.
    public var layerStatistics: [Int: GPUPagedKVPool.Statistics] {
        get throws { try pools.mapValues { try $0.statistics } }
    }

    public var reservedArenaBytes: Int { pools.values.reduce(0) { $0 + $1.admittedBytes } }
    public var outstandingDecodeClaimPages: Int { pageAdmission.statistics.claimedPages }
    public var pageAdmissionStatistics: QwenPagedPageAdmission.Statistics { pageAdmission.statistics }

    /// Call only under this model's exclusive inference gate. The returned
    /// constant claim covers the entire remaining request, not one suffix.
    func reserveDecodePages(pagesPerLayer: Int) throws -> QwenPagedDecodePageLease? {
        try pageAdmission.reserveDecodePages(pagesPerLayer: pagesPerLayer,
            minimumFreePages: minimumHealthyFreePages())
    }

    /// Preflight immediately before optional attachment creation, without
    /// yielding or callbacks before its last per-layer native allocation.
    func canReserveOptionalPages(pagesPerLayer: Int) throws -> Bool {
        try pageAdmission.canReserveOptionalPages(pagesPerLayer: pagesPerLayer,
            minimumFreePages: minimumHealthyFreePages())
    }

    private func minimumHealthyFreePages() throws -> Int {
        let statistics = try layerStatistics
        guard Set(statistics.keys) == Set(stride(from: 3, to: 48, by: 4)),
              statistics.values.allSatisfy({ $0.physicalPages == UInt64(maximumPagesPerLayer) &&
                  $0.failedOperations == 0 && $0.freePages + $0.livePages == $0.physicalPages }),
              let free = statistics.values.map(\.freePages).min() else {
            throw GPUError.invalid("Physical KV admission requires healthy complete model pools")
        }
        return Int(free)
    }
}
