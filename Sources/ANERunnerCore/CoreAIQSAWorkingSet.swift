import Foundation

/// Optional prefill functions compute attention over a bounded working view;
/// their input/output state tensors retain the full model capacity.
struct CoreAIQSAWorkingSet: Decodable, Equatable, Sendable {
    enum ValidationError: LocalizedError {
        case invalidEntries
        var errorDescription: String? { "Invalid QSA prefill working-set functions" }
    }
    let tokenCount: Int
    let kvLimit: Int
    let function: String

    static func validate(_ entries: [Self], counts: [Int], capacity: Int) throws {
        guard Set(entries.map(\.function)).count == entries.count,
              entries.allSatisfy({ entry in
                  entry.tokenCount > 1 && counts.contains(entry.tokenCount)
                      && entry.kvLimit >= entry.tokenCount && entry.kvLimit < capacity
                      && entry.kvLimit.isMultiple(of: 4)
                      && entry.function == "prefill_s\(entry.tokenCount)_kv\(entry.kvLimit)"
              }) else {
            throw ValidationError.invalidEntries
        }
    }

    static func select(_ entries: [Self], count: Int, endOffset: Int) -> Self? {
        guard count > 1, endOffset >= count else { return nil }
        return entries.filter { $0.tokenCount == count && endOffset <= $0.kvLimit }
            .min { $0.kvLimit < $1.kvLimit }
    }
}
