import Foundation

/// Explicit service admission limits. Body bytes, logical tokens and connection
/// lifetime are independent quotas; none estimates model activation memory.
/// Existing HTTP defaults remain unchanged when these options are omitted.
public struct QwenHTTPServiceCapacity: Encodable, Equatable, Sendable {
    public enum ValidationError: Error, LocalizedError, Equatable {
        case invalid(String)
        public var errorDescription: String? {
            switch self { case .invalid(let reason): return "HTTP service capacity: " + reason }
        }
    }

    public static let maximumContextLimit = 262_144
    public static let maximumBodyBytes = 64 * 1024 * 1024
    public let contextLimit, maxReservedTokens, maxResidentSequences: Int
    public let maxBodyBytes, connectionDeadlineSeconds: Int

    public init(contextLimit: Int = 16_384, maxReservedTokens: Int = 32_768,
                maxResidentSequences: Int = 2, maxBodyBytes: Int = 262_144,
                connectionDeadlineSeconds: Int = 300) throws {
        guard (1...Self.maximumContextLimit).contains(contextLimit) else {
            throw ValidationError.invalid("--context-limit must be in 1...262144")
        }
        guard maxReservedTokens >= contextLimit else {
            throw ValidationError.invalid("--max-reserved-tokens must be at least --context-limit")
        }
        // This change supports a single large resident and preserves the
        // existing two-resident option; it does not expand HTTP concurrency.
        guard (1...2).contains(maxResidentSequences) else {
            throw ValidationError.invalid("--max-resident-sequences must be in 1...2")
        }
        guard (1024...Self.maximumBodyBytes).contains(maxBodyBytes) else {
            throw ValidationError.invalid("--max-body-bytes must be in 1024...67108864")
        }
        guard (1...86_400).contains(connectionDeadlineSeconds) else {
            throw ValidationError.invalid("--connection-deadline-seconds must be in 1...86400")
        }
        self.contextLimit = contextLimit; self.maxReservedTokens = maxReservedTokens
        self.maxResidentSequences = maxResidentSequences; self.maxBodyBytes = maxBodyBytes
        self.connectionDeadlineSeconds = connectionDeadlineSeconds
    }

    /// Recheck the loaded model before advertising ready. The dedicated native
    /// kernels still impose their own262144-position bound independently.
    public func validateModelMaximumPositions(_ maximumPositions: Int) throws {
        guard maximumPositions > 0, contextLimit <= maximumPositions else {
            throw ValidationError.invalid("--context-limit exceeds the loaded model's maximum positions")
        }
    }
}
