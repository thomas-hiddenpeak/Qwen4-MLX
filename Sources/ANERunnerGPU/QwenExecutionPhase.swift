/// Business phase of a trunk forward, independent of its tensor shape and the
/// selected experimental kernels. Callers should pass this explicitly.
public enum QwenExecutionPhase: String, Codable, CaseIterable, Sendable {
    case prefill
    case decode
    case verification

    /// Compatibility for older low-level callers only: the former policy used
    /// S1 as decode and any longer input as prefill. It cannot identify MTP from
    /// shape alone, so MTP and high-level generation must supply an explicit phase.
    static func resolve(_ requested: Self?, tokenCount: Int) throws -> Self {
        guard tokenCount > 0 else { throw GPUError.invalid("Execution phase requires at least one input token") }
        let phase = requested ?? (tokenCount == 1 ? .decode : .prefill)
        switch phase {
        case .prefill:
            break
        case .decode:
            guard tokenCount == 1 else { throw GPUError.invalid("Decode phase requires one input token") }
        case .verification:
            guard tokenCount <= 5 else { throw GPUError.invalid("Native MTP verification supports one through five input tokens") }
        }
        return phase
    }

    /// Retain the existing schedule: a single-token call relies on the caller's
    /// final evaluation; multi-token prefill and verification may materialize
    /// every N layers. The supplied interval belongs to this forward's phase,
    /// so tuning verification does not change prefill or autoregressive decode.
    func shouldEvaluate(completedLayers: Int, tokenCount: Int, every interval: Int) -> Bool {
        guard tokenCount > 1, completedLayers > 0, interval > 0 else { return false }
        switch self {
        case .decode:
            return false
        case .prefill, .verification:
            return completedLayers % interval == 0
        }
    }

    static func validateVerificationInterval(_ interval: Int) throws {
        guard (1...48).contains(interval) else {
            throw GPUError.invalid("MTP verification evaluation interval must be within 1...48 layers")
        }
    }
}
