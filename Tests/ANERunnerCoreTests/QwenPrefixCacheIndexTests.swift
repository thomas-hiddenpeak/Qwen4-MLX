import XCTest
@testable import ANERunnerCore

final class QwenPrefixCacheIndexTests: XCTestCase {
    typealias Cache = QwenPrefixCacheIndex<String>

    private func cache(entries: Int = 16, bytes: Int = 1_000,
                       tokens: Int = 1_000) throws -> Cache {
        try Cache(maxEntries: entries, maxBytes: bytes, maxKeyTokens: tokens)
    }

    @discardableResult
    private func put(_ cache: Cache, _ tokens: [Int32], _ value: String,
                     namespace: String = "model-a", bytes: Int = 1) -> Bool {
        cache.insert(tokens: tokens, namespace: namespace, value: value,
                     logicalPayloadBytes: bytes)
    }

    private func value(_ cache: Cache, _ tokens: [Int32], cap: Int? = nil,
                       namespace: String = "model-a") -> String? {
        cache.peek(tokens: tokens, namespace: namespace, maxPrefixTokens: cap)?.value
    }

    func testLimitsMustBePositiveAndIntMaxDoesNotOverflowAccounting() throws {
        for (entries, bytes, tokens) in [(0, 1, 1), (-1, 1, 1), (1, 0, 1),
                                         (1, -1, 1), (1, 1, 0), (1, 1, -1)] {
            XCTAssertThrowsError(try cache(entries: entries, bytes: bytes, tokens: tokens)) {
                XCTAssertEqual($0 as? Cache.ConfigurationError, .invalidLimits)
            }
        }
        let c = try cache(entries: Int.max, bytes: Int.max, tokens: Int.max)
        XCTAssertTrue(put(c, [1], "full", bytes: Int.max))
        XCTAssertTrue(put(c, [2], "next", bytes: 1))
        XCTAssertNil(value(c, [1]))
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 1)
        XCTAssertEqual(c.statistics.evictions, 1)
    }

    func testLongestStoredBoundaryAndPartialRadixEdges() throws {
        let c = try cache()
        put(c, [10, 20, 30, 40], "left")
        put(c, [10, 20, 31, 41], "right")
        // [10,20] is a branching node, but no state was saved there.
        XCTAssertNil(value(c, [10, 20]))
        XCTAssertNil(value(c, [10, 20, 30]))
        XCTAssertNil(value(c, [10, 20, 30, 99]))
        put(c, [10, 20], "saved-ancestor")
        XCTAssertEqual(value(c, [10, 20, 30]), "saved-ancestor")
        XCTAssertEqual(value(c, [10, 20, 30, 99]), "saved-ancestor")
        XCTAssertEqual(value(c, [10, 20, 30, 40, 50]), "left")
        XCTAssertEqual(value(c, [10, 20, 31, 41, 50]), "right")
        XCTAssertEqual(value(c, [10, 20, 30, 40], cap: 3), "saved-ancestor")
        XCTAssertEqual(value(c, [10, 20, 30, 40], cap: 4), "left")
        XCTAssertEqual(value(c, [10, 20, 30, 40], cap: Int.max), "left")
        for cap in [Int.min, -1, 0, 1] {
            XCTAssertNil(value(c, [10, 20, 30, 40], cap: cap))
        }
        XCTAssertNil(value(c, []))
        let match = try XCTUnwrap(c.peek(tokens: [10, 20, 30, 40, 50], namespace: "model-a"))
        XCTAssertEqual(match.prefixTokenCount, 4)
        XCTAssertEqual(match.logicalPayloadBytes, 1)
    }

    func testNamespaceAndSignedTokenIdentityAreExact() throws {
        let c = try cache()
        let key: [Int32] = [Int32.min, -1, 0, Int32.max]
        put(c, key, "first", namespace: "weights:1/settings:1")
        put(c, key, "other", namespace: "weights:2/settings:1")
        XCTAssertEqual(value(c, key, namespace: "weights:1/settings:1"), "first")
        XCTAssertEqual(value(c, key, namespace: "weights:2/settings:1"), "other")
        XCTAssertNil(value(c, key, namespace: "weights:1/settings:2"))
        XCTAssertNil(value(c, [Int32.min, -1, 1, Int32.max], namespace: "weights:1/settings:1"))
        XCTAssertTrue(c.remove(tokens: key, namespace: "weights:1/settings:1"))
        XCTAssertEqual(value(c, key, namespace: "weights:2/settings:1"), "other")
        XCTAssertEqual(c.statistics.entries, 1)
        XCTAssertEqual(c.statistics.keyTokens, 4)
    }

    func testTouchAndGlobalLRUEvictOnlyLeastRecentlyUsedEntry() throws {
        let c = try cache(entries: 2)
        put(c, [1, 2], "first")
        put(c, [3], "second", namespace: "other")
        XCTAssertEqual(c.lookup(tokens: [1, 2, 9], namespace: "model-a")?.value, "first")
        put(c, [4], "third")
        XCTAssertEqual(value(c, [1, 2]), "first")
        XCTAssertNil(value(c, [3], namespace: "other"))
        XCTAssertEqual(value(c, [4]), "third")
        XCTAssertEqual(c.statistics.hits, 1)
        XCTAssertEqual(c.statistics.evictions, 1)
        XCTAssertEqual(c.statistics.entries, 2)
    }

    func testPeekPreservesCountersAndNoTouchLookupPreservesLRU() throws {
        let c = try cache(entries: 2)
        put(c, [1], "oldest"); put(c, [2], "newest")
        let before = c.statistics
        XCTAssertEqual(value(c, [1]), "oldest")
        XCTAssertNil(value(c, [99]))
        XCTAssertEqual(c.statistics, before)
        XCTAssertEqual(c.lookup(tokens: [1], namespace: "model-a", touch: false)?.value, "oldest")
        XCTAssertNil(c.lookup(tokens: [99], namespace: "model-a", touch: false))
        put(c, [3], "third")
        XCTAssertNil(value(c, [1]))
        XCTAssertEqual(value(c, [2]), "newest")
        XCTAssertEqual(c.statistics.hits, 1)
        XCTAssertEqual(c.statistics.misses, 1)
        XCTAssertEqual(c.statistics.evictions, 1)
    }

    func testBytePressureCanEvictMultipleEntries() throws {
        let c = try cache(bytes: 10)
        put(c, [1], "a", bytes: 3); put(c, [2], "b", bytes: 3)
        put(c, [3], "c", bytes: 3)
        XCTAssertTrue(put(c, [4], "large", bytes: 8))
        XCTAssertEqual(c.statistics.entries, 1)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 8)
        XCTAssertEqual(c.statistics.keyTokens, 1)
        XCTAssertEqual(c.statistics.evictions, 3)
        for key: Int32 in [1, 2, 3] { XCTAssertNil(value(c, [key])) }
    }

    func testKeyTokenBudgetChargesSharedPrefixesAndEvictsIndependently() throws {
        let c = try cache(tokens: 7)
        put(c, [1, 2, 3], "a"); put(c, [1, 2, 4], "b")
        XCTAssertEqual(c.statistics.keyTokens, 6)
        _ = c.lookup(tokens: [1, 2, 3], namespace: "model-a")
        put(c, [7, 8], "c")
        XCTAssertEqual(value(c, [1, 2, 3]), "a")
        XCTAssertNil(value(c, [1, 2, 4]))
        XCTAssertEqual(c.statistics.entries, 2)
        XCTAssertEqual(c.statistics.keyTokens, 5)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 2)
        XCTAssertEqual(c.statistics.evictions, 1)
    }

    func testReplacementReaccountsAndBecomesMostRecent() throws {
        let c = try cache(entries: 3, bytes: 10)
        put(c, [1, 2], "a", bytes: 3); put(c, [3, 4], "b", bytes: 4)
        put(c, [5], "c", bytes: 2)
        XCTAssertTrue(put(c, [1, 2], "a-new", bytes: 8))
        XCTAssertEqual(value(c, [1, 2]), "a-new")
        XCTAssertNil(value(c, [3, 4]))
        XCTAssertEqual(value(c, [5]), "c")
        XCTAssertEqual(c.statistics.entries, 2)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 10)
        XCTAssertEqual(c.statistics.keyTokens, 3)
        XCTAssertEqual(c.statistics.evictions, 1)
        put(c, [6], "d", bytes: 1)
        XCTAssertNil(value(c, [5]))
        XCTAssertEqual(value(c, [1, 2]), "a-new")
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 9)
        // Updating to a smaller value releases its old byte charge exactly.
        put(c, [1, 2], "a-small", bytes: 0)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 1)
        XCTAssertEqual(c.statistics.entries, 2)
        XCTAssertEqual(c.statistics.keyTokens, 3)
    }

    func testRejectedInsertsDoNotPolluteExistingStateOrRecency() throws {
        let c = try cache(entries: 2, bytes: 5, tokens: 5)
        put(c, [1], "oldest"); put(c, [2], "newest")
        let before = c.statistics
        XCTAssertFalse(put(c, [1], "oversized-replacement", bytes: 6))
        XCTAssertFalse(put(c, [1], "negative-replacement", bytes: -1))
        XCTAssertFalse(put(c, [], "empty"))
        XCTAssertFalse(put(c, [1, 2, 3, 4, 5, 6], "too-many-key-tokens", namespace: "never-created"))
        XCTAssertEqual(c.statistics, before)
        XCTAssertEqual(value(c, [1]), "oldest")
        put(c, [3], "third")
        XCTAssertNil(value(c, [1]))
        XCTAssertEqual(value(c, [2]), "newest")
        XCTAssertEqual(c.statistics.evictions, 1)
    }

    func testRemovalRecompressesBranchesWithoutRemovingAncestorOrSibling() throws {
        let c = try cache()
        put(c, [1], "root"); put(c, [1, 2], "middle")
        put(c, [1, 2, 3, 4], "left"); put(c, [1, 2, 3, 5], "right")
        XCTAssertFalse(c.remove(tokens: [1, 2, 3], namespace: "model-a"))
        XCTAssertFalse(c.remove(tokens: [1, 2, 3, 4, 5], namespace: "model-a"))
        XCTAssertTrue(c.remove(tokens: [1, 2], namespace: "model-a"))
        XCTAssertEqual(value(c, [1, 2, 3]), "root")
        XCTAssertTrue(c.remove(tokens: [1, 2, 3, 4], namespace: "model-a"))
        XCTAssertEqual(value(c, [1, 2, 3, 5]), "right")
        XCTAssertEqual(value(c, [1, 2, 3, 4]), "root")
        put(c, [1, 2, 3], "new-middle")
        XCTAssertEqual(value(c, [1, 2, 3, 4]), "new-middle")
        XCTAssertEqual(value(c, [1, 2, 3, 5]), "right")
        XCTAssertTrue(c.remove(tokens: [1], namespace: "model-a"))
        XCTAssertTrue(c.remove(tokens: [1, 2, 3], namespace: "model-a"))
        XCTAssertTrue(c.remove(tokens: [1, 2, 3, 5], namespace: "model-a"))
        XCTAssertFalse(c.remove(tokens: [1, 2, 3, 5], namespace: "model-a"))
        XCTAssertEqual(c.statistics.entries, 0)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 0)
        XCTAssertEqual(c.statistics.keyTokens, 0)
        XCTAssertEqual(c.statistics.evictions, 0)
        put(c, [1, 2, 3, 5], "recreated")
        XCTAssertEqual(value(c, [1, 2, 3, 5]), "recreated")
    }

    func testClearAndCodableStatistics() throws {
        let c = try cache(entries: 1)
        put(c, [1], "a"); put(c, [2, 3], "b")
        _ = c.lookup(tokens: [2, 3], namespace: "model-a")
        _ = c.lookup(tokens: [9], namespace: "model-a")
        let decoded = try JSONDecoder().decode(QwenPrefixCacheStatistics.self,
            from: JSONEncoder().encode(c.statistics))
        XCTAssertEqual(decoded, c.statistics)
        c.clear()
        XCTAssertEqual(c.statistics.entries, 0)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 0)
        XCTAssertEqual(c.statistics.keyTokens, 0)
        XCTAssertEqual(c.statistics.hits, 1)
        XCTAssertEqual(c.statistics.misses, 1)
        XCTAssertEqual(c.statistics.evictions, 1)
        XCTAssertNil(value(c, [2, 3]))
        c.clear(resetStatistics: true)
        XCTAssertEqual(c.statistics.hits, 0)
        XCTAssertEqual(c.statistics.misses, 0)
        XCTAssertEqual(c.statistics.evictions, 0)
    }

    private final class Payload {}

    func testCacheReleasesValuesButOutstandingMatchOwnsItsSnapshot() throws {
        let c = try QwenPrefixCacheIndex<Payload>(maxEntries: 1, maxBytes: 10)
        var original: Payload? = Payload()
        weak var weakOriginal = original
        c.insert(tokens: [1], namespace: "a", value: original!, logicalPayloadBytes: 1)
        original = nil
        XCTAssertNotNil(weakOriginal)
        var held = c.lookup(tokens: [1], namespace: "a")
        XCTAssertNotNil(held?.value)
        c.insert(tokens: [2], namespace: "b", value: Payload(), logicalPayloadBytes: 2)
        XCTAssertNotNil(weakOriginal)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 2)
        held = nil
        XCTAssertNil(weakOriginal)

        var replacement: Payload? = Payload()
        weak var weakReplacement = replacement
        c.insert(tokens: [2], namespace: "b", value: replacement!, logicalPayloadBytes: 3)
        replacement = nil
        XCTAssertNotNil(weakReplacement)
        XCTAssertTrue(c.remove(tokens: [2], namespace: "b"))
        XCTAssertNil(weakReplacement)

        var cleared: Payload? = Payload()
        weak var weakCleared = cleared
        c.insert(tokens: [3], namespace: "c", value: cleared!, logicalPayloadBytes: 4)
        cleared = nil
        c.clear()
        XCTAssertNil(weakCleared)
    }

    func testReplacementAndCacheDestructionDoNotRetainValues() throws {
        var c: QwenPrefixCacheIndex<Payload>? = try .init(maxEntries: 3, maxBytes: 10)
        var old: Payload? = Payload()
        weak var weakOld = old
        c!.insert(tokens: [1], namespace: "a", value: old!, logicalPayloadBytes: 1)
        old = nil
        c!.insert(tokens: [1], namespace: "a", value: Payload(), logicalPayloadBytes: 1)
        XCTAssertNil(weakOld)
        var first: Payload? = Payload(), second: Payload? = Payload()
        weak var weakFirst = first
        weak var weakSecond = second
        c!.insert(tokens: [1, 2], namespace: "a", value: first!, logicalPayloadBytes: 1)
        c!.insert(tokens: [1, 3], namespace: "a", value: second!, logicalPayloadBytes: 1)
        first = nil; second = nil
        c = nil
        XCTAssertNil(weakFirst)
        XCTAssertNil(weakSecond)
    }

    func testEvictingAncestorPreservesDescendantAndReleasesOnlyAncestor() throws {
        let c = try QwenPrefixCacheIndex<Payload>(maxEntries: 2, maxBytes: 10)
        var ancestor: Payload? = Payload(), descendant: Payload? = Payload()
        weak var weakAncestor = ancestor
        weak var weakDescendant = descendant
        c.insert(tokens: [1], namespace: "a", value: ancestor!, logicalPayloadBytes: 1)
        c.insert(tokens: [1, 2], namespace: "a", value: descendant!, logicalPayloadBytes: 2)
        ancestor = nil; descendant = nil
        c.insert(tokens: [3], namespace: "a", value: Payload(), logicalPayloadBytes: 3)
        XCTAssertNil(weakAncestor)
        XCTAssertNotNil(weakDescendant)
        XCTAssertNil(c.peek(tokens: [1], namespace: "a"))
        XCTAssertTrue(c.peek(tokens: [1, 2, 9], namespace: "a")?.value === weakDescendant)
        XCTAssertEqual(c.statistics.logicalPayloadBytes, 5)
        XCTAssertEqual(c.statistics.keyTokens, 3)
        c.clear()
        XCTAssertNil(weakDescendant)
    }

    func testDeterministicRadixMutationsMatchFlatExactPrefixOracle() throws {
        let c = try cache(entries: 200, bytes: 200, tokens: 1_000)
        var keys: [[Int32]] = []
        func enumerate(_ prefix: [Int32], remaining: Int) {
            if !prefix.isEmpty { keys.append(prefix) }
            if remaining > 0 {
                for token: Int32 in [-1, 0, 1] { enumerate(prefix + [token], remaining: remaining - 1) }
            }
        }
        enumerate([], remaining: 4)
        // Descendants first forces splitting compressed edges on later inserts.
        var expected: [[Int32]: String] = [:]
        for (index, key) in keys.reversed().enumerated() where index % 3 != 0 {
            let label = "entry-\(index)"
            put(c, key, label); expected[key] = label
        }
        func compareAll() {
            for query in keys {
                for cap in 0...4 {
                    let best = expected.keys.filter { $0.count <= cap && query.starts(with: $0) }
                        .max { $0.count < $1.count }
                    XCTAssertEqual(value(c, query, cap: cap), best.flatMap { expected[$0] },
                                   "query=\(query), cap=\(cap)")
                }
            }
            XCTAssertEqual(c.statistics.entries, expected.count)
            XCTAssertEqual(c.statistics.keyTokens, expected.keys.reduce(0) { $0 + $1.count })
            XCTAssertEqual(c.statistics.logicalPayloadBytes, expected.count)
        }
        compareAll()
        for (index, key) in keys.enumerated() where index % 2 == 0 {
            XCTAssertEqual(c.remove(tokens: key, namespace: "model-a"), expected.removeValue(forKey: key) != nil)
        }
        compareAll()
        for (index, key) in keys.enumerated() where index % 5 == 0 {
            let label = "replacement-\(index)"
            put(c, key, label); expected[key] = label
        }
        compareAll()
    }
}
