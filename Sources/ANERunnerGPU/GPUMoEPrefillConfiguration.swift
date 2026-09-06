import Foundation

/// Explicit request-local prefill selection. Unmeasured lengths retain the
/// reference reduction. The optional fused gate/up applies to multi-token chunks.
/// Neither optimization is consulted by decode or MTP verification.
public struct GPUMoEPrefillConfiguration: Codable, Equatable, Sendable {
    public let version: Int
    public let threadgroups: [Int: Int]
    public let gateUpVariant: Int?
    public let groupedDown: Bool?

    public init(threadgroups: [Int: Int], gateUpVariant: Int? = nil, groupedDown: Bool? = nil) {
        self.version = 1
        self.threadgroups = threadgroups
        self.gateUpVariant = gateUpVariant
        self.groupedDown = groupedDown
    }

    public func validated(phase: QwenExecutionPhase = .prefill) throws {
        guard phase == .prefill, version == 1,
              gateUpVariant == nil || (0...3).contains(gateUpVariant!),
              groupedDown != true || (gateUpVariant.map { $0 >= 2 } ?? false),
              threadgroups.keys.allSatisfy({ (2...512).contains($0) }),
              threadgroups.values.allSatisfy({ [128,256,512].contains($0) }) else {
            throw GPUError.invalid("Unsupported prefill MoE configuration")
        }
    }

    public func threadgroupSize(tokenCount: Int) -> Int? { threadgroups[tokenCount] }

    /// Match the stock sorted QMM batch threshold for the grouped candidates.
    /// Short tails retain the reference kernel and its original accumulation.
    public func effectiveGateUpVariant(tokenCount: Int) -> Int? {
        guard let variant = gateUpVariant else { return nil }
        return ((variant >= 2 ? 205 : 2)...512).contains(tokenCount) ? variant : nil
    }

    public func usesGroupedDown(tokenCount: Int) -> Bool {
        groupedDown == true && (effectiveGateUpVariant(tokenCount: tokenCount).map { $0 >= 2 } ?? false)
    }
}
