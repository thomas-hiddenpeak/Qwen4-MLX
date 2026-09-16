import Foundation

/// Errors shared by CPU model metadata and tokenization paths.
public enum QwenModelDataError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String { switch self { case .invalid(let message): message } }
}

public enum QwenGenerationError: Error, LocalizedError, Equatable {
    case invalidRequest(String)
    case busy
    case resourceLimit(String)
    case cancelled
    case unavailable(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let reason): return "Invalid generation request: \(reason)"
        case .resourceLimit(let reason): return "State capacity unavailable: \(reason)"
        case .busy: return "This model is already processing a generation request"
        case .cancelled: return "Generation was cancelled"
        case .unavailable(let reason): return "Generation is unavailable: \(reason)"
        }
    }
}
