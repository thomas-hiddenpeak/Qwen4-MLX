/// Per-request HTTP adapter state, owned by the inference thread. This records
/// only byte counts and a finite local failure reason; never text or errors.
/// The separate QwenSSEOutputBuffer still chooses the transport outcome.
public struct QwenHTTPTextBudget: Sendable {
    public enum Failure: String, Sendable {
        case textLimit = "text_limit"
        case responseLimit = "response_limit"
        case encodingFailed = "encoding_failed"

        public var code: String {
            self == .encodingFailed ? "encoding_failed" : "output_limit"
        }
        public var message: String {
            self == .encodingFailed ? "Response could not be encoded" : "Output exceeds configured byte limit"
        }
    }
    public let maxBytes: Int
    public private(set) var acceptedBytes = 0
    public private(set) var failure: Failure?

    public init(maxBytes: Int) {
        precondition(maxBytes >= 0)
        self.maxBytes = maxBytes
    }

    /// Charge only accepted UTF-8 text. The caller retains the actual text.
    public mutating func accept(byteCount: Int) -> Bool {
        precondition(byteCount >= 0)
        guard failure == nil else { return false }
        guard byteCount <= maxBytes - acceptedBytes else {
            record(.textLimit)
            return false
        }
        acceptedBytes += byteCount
        return true
    }

    /// Preserve the first local adapter failure. This does not override an
    /// earlier SSE overflow, disconnect, or other buffer terminal selection.
    public mutating func record(_ reason: Failure) {
        if failure == nil { failure = reason }
    }

    public func errorResponse(cancelled: Bool) -> (code: String, message: String) {
        if let failure { return (failure.code, failure.message) }
        return cancelled ? ("cancelled", "Generation cancelled") : ("generation_failed", "Generation failed")
    }
}
