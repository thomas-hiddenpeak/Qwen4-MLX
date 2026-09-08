import Foundation

/// A shared logical-byte admission ledger for request state, retained cache
/// entries and temporary state copies. It does not measure process RSS or MLX
/// allocations: callers reserve before creating storage and retain each lease
/// for exactly as long as they retain the corresponding storage.
///
/// Every operation, including lease destruction, is synchronized by one lock.
/// Leases retain the ledger, while the ledger never retains its leases.
public final class QwenStateBudget: @unchecked Sendable {
    public enum Kind: String, Codable, CaseIterable, Sendable {
        case request, cache, workspace
    }

    public enum ConfigurationError: Error, Equatable {
        case invalidMaxBytes
    }

    public struct Statistics: Codable, Equatable, Sendable {
        public let maxBytes: Int
        public let requestBytes, cacheBytes, workspaceBytes: Int
        public let totalBytes, peakBytes: Int
        /// Denied reserve/resize attempts, including invalid sizes and attempts
        /// to resize a released lease. Saturates at Int.max.
        public let rejections: Int
        public let currentLeases: Int
    }

    /// Owns one reservation. Multiple references to a lease still represent one
    /// reservation; explicit release and deinit are both safe on any thread.
    public final class Lease: @unchecked Sendable {
        public let kind: Kind
        private let budget: QwenStateBudget
        fileprivate var reservedBytes: Int
        fileprivate var released = false

        fileprivate init(budget: QwenStateBudget, bytes: Int, kind: Kind) {
            self.budget = budget; reservedBytes = bytes; self.kind = kind
        }

        /// Current reservation, or zero after release.
        public var bytes: Int { budget.withLock { reservedBytes } }
        public var isReleased: Bool { budget.withLock { released } }

        /// Grow only if the complete delta fits; shrinking releases the delta
        /// immediately. A failed resize leaves the reservation unchanged.
        /// Sizes must be positive. A released lease cannot be revived.
        @discardableResult
        public func resize(to bytes: Int) -> Bool { budget.resize(self, to: bytes) }

        /// Idempotent. Storage owned by the caller must no longer rely on this
        /// reservation when it is explicitly released.
        public func release() { budget.release(self) }
        deinit { release() }
    }

    public let maxBytes: Int
    private let lock = NSLock()
    private var requestBytes = 0, cacheBytes = 0, workspaceBytes = 0
    private var totalBytes = 0, peakBytes = 0, rejections = 0, currentLeases = 0

    public init(maxBytes: Int) throws {
        guard maxBytes > 0 else { throw ConfigurationError.invalidMaxBytes }
        self.maxBytes = maxBytes
    }

    public var statistics: Statistics {
        withLock {
            .init(maxBytes: maxBytes, requestBytes: requestBytes, cacheBytes: cacheBytes,
                  workspaceBytes: workspaceBytes, totalBytes: totalBytes,
                  peakBytes: peakBytes, rejections: rejections, currentLeases: currentLeases)
        }
    }

    /// Reserve atomically or return nil without charging any bytes. Reservation
    /// sizes must be positive. This method never blocks waiting for capacity.
    public func reserve(bytes: Int, kind: Kind) -> Lease? {
        withLock {
            // Subtract first: even an Int.max configuration cannot overflow.
            guard bytes > 0, bytes <= maxBytes - totalBytes else {
                reject(); return nil
            }
            changeBytes(by: bytes, kind: kind)
            // Each live lease owns at least one byte, so count <= totalBytes.
            currentLeases += 1
            return Lease(budget: self, bytes: bytes, kind: kind)
        }
    }

    private func resize(_ lease: Lease, to bytes: Int) -> Bool {
        withLock {
            guard !lease.released, bytes > 0 else { reject(); return false }
            // Both sizes are positive, so this subtraction cannot overflow.
            let delta = bytes - lease.reservedBytes
            guard delta <= maxBytes - totalBytes else { reject(); return false }
            changeBytes(by: delta, kind: lease.kind)
            lease.reservedBytes = bytes
            return true
        }
    }

    private func release(_ lease: Lease) {
        withLock {
            guard !lease.released else { return }
            changeBytes(by: -lease.reservedBytes, kind: lease.kind)
            currentLeases -= 1
            lease.reservedBytes = 0
            lease.released = true
        }
    }

    /// Called only with the lock held, after validating a growth delta. All
    /// category counters remain nonnegative and no larger than totalBytes.
    private func changeBytes(by delta: Int, kind: Kind) {
        switch kind {
        case .request: requestBytes += delta
        case .cache: cacheBytes += delta
        case .workspace: workspaceBytes += delta
        }
        totalBytes += delta
        peakBytes = max(peakBytes, totalBytes)
    }

    private func reject() { if rejections < Int.max { rejections += 1 } }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}
