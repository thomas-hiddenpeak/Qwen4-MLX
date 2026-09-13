import Foundation

/// Conservative forecast of future physical KV allocations. Existing occupied
/// slots are measured separately by the native pool. A decode claim stays at
/// its original size even after some slots materialize, deliberately counting
/// them twice when admitting another request or an optional cache attachment.
public final class QwenPagedPageAdmission: @unchecked Sendable {
    public enum AdmissionError: Error, Equatable {
        case invalidCapacity, invalidPageCount, invalidFreePages, invalidTokenRange
    }
    public struct Statistics: Codable, Equatable, Sendable {
        public let maximumPagesPerLayer, claimedPages, activeClaims, deniedClaims: Int
    }
    public final class Lease: @unchecked Sendable {
        public let pagesPerLayer: Int
        private let ledger: QwenPagedPageAdmission
        fileprivate var released = false
        fileprivate init(ledger: QwenPagedPageAdmission, pagesPerLayer: Int) {
            self.ledger = ledger; self.pagesPerLayer = pagesPerLayer
        }
        public var isReleased: Bool { ledger.withLock { released } }
        /// Abandon/finish the cursor and join its pending work before release.
        /// Never release merely because the first suffix import has completed.
        public func release() { ledger.release(self) }
        deinit { release() }
    }

    public let maximumPagesPerLayer: Int
    private let lock = NSLock()
    private var claimedPages = 0, activeClaims = 0, deniedClaims = 0

    public init(maximumPagesPerLayer: Int) throws {
        guard (1...4096).contains(maximumPagesPerLayer) else { throw AdmissionError.invalidCapacity }
        self.maximumPagesPerLayer = maximumPagesPerLayer
    }
    public var statistics: Statistics {
        withLock { .init(maximumPagesPerLayer: maximumPagesPerLayer,
            claimedPages: claimedPages, activeClaims: activeClaims, deniedClaims: deniedClaims) }
    }

    /// End excludes the last emitted token, which is never forwarded. Complete
    /// ancestor pages are shared; its partial tail remains pinned separately.
    /// One additional slot permits an immutable COW old/new tail to coexist.
    public static func requiredDecodePages(promptTokens: Int, maximumOutputTokens: Int,
                                           reusedPrefixTokens: Int) throws -> Int {
        guard promptTokens > 0, maximumOutputTokens > 0,
              reusedPrefixTokens >= 0, reusedPrefixTokens <= promptTokens else {
            throw AdmissionError.invalidTokenRange
        }
        let (end, overflow) = promptTokens.addingReportingOverflow(maximumOutputTokens - 1)
        guard !overflow else { throw AdmissionError.invalidTokenRange }
        let finalPages = end / 32 + (end % 32 == 0 ? 0 : 1)
        return finalPages - reusedPrefixTokens / 32 + 1
    }

    /// minimumFreePages is the minimum across every model attention pool,
    /// sampled under the inference executor's exclusive gate. Only native
    /// completion reclamation may run concurrently; it can only increase free.
    public func reserveDecodePages(pagesPerLayer: Int, minimumFreePages: Int) throws -> Lease? {
        try validate(pagesPerLayer: pagesPerLayer, minimumFreePages: minimumFreePages, allowZero: false)
        return withLock {
            guard pagesPerLayer <= maximumPagesPerLayer,
                  claimedPages <= minimumFreePages,
                  pagesPerLayer <= minimumFreePages - claimedPages else {
                if deniedClaims < Int.max { deniedClaims += 1 }
                return nil
            }
            claimedPages += pagesPerLayer; activeClaims += 1
            return Lease(ledger: self, pagesPerLayer: pagesPerLayer)
        }
    }

    /// A zero-row/equal-offset fork allocates no physical slot and may proceed
    /// even when conservative outstanding claims exceed current free pages.
    public func canReserveOptionalPages(pagesPerLayer: Int, minimumFreePages: Int) throws -> Bool {
        try validate(pagesPerLayer: pagesPerLayer, minimumFreePages: minimumFreePages, allowZero: true)
        return withLock {
            pagesPerLayer == 0 || (claimedPages <= minimumFreePages &&
                pagesPerLayer <= minimumFreePages - claimedPages)
        }
    }
    private func validate(pagesPerLayer: Int, minimumFreePages: Int, allowZero: Bool) throws {
        guard pagesPerLayer >= (allowZero ? 0 : 1) else { throw AdmissionError.invalidPageCount }
        guard (0...maximumPagesPerLayer).contains(minimumFreePages) else { throw AdmissionError.invalidFreePages }
    }
    private func release(_ lease: Lease) {
        withLock {
            guard !lease.released else { return }
            claimedPages -= lease.pagesPerLayer; activeClaims -= 1; lease.released = true
        }
    }
    private func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock(); defer { lock.unlock() }
        return try body()
    }
}
