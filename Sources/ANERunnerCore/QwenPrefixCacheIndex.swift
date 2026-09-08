/// Accounting for entries owned by a prefix cache. Payload bytes are supplied
/// by the caller; they are neither physical GPU allocations nor process RSS.
public struct QwenPrefixCacheStatistics: Codable, Equatable, Sendable {
    public let hits: Int
    public let misses: Int
    public let evictions: Int
    public let entries: Int
    public let logicalPayloadBytes: Int
    /// Sum of complete token-key lengths, including shared prefixes once per
    /// entry. Radix edge storage is separately bounded by this same total.
    public let keyTokens: Int
    /// Runtime snapshot counters, populated by the model-specific cache owner.
    public var published: Int = 0
    public var skippedOversize: Int = 0
    public var restoreFailures: Int = 0
}

/// A strong reference to a saved value at an exact, complete token boundary.
/// Removing/evicting an entry releases the cache's ownership, but this result
/// or other outside references may keep the value alive. Such references are
/// not included in the cache's current byte accounting.
public struct QwenPrefixCacheMatch<Value> {
    public let prefixTokenCount: Int
    public let value: Value
    public let logicalPayloadBytes: Int
}

/// A bounded, namespace-isolated radix index for complete prefix snapshots.
///
/// Only inserted token boundaries are hits: an internal branching node is not
/// a checkpoint. Namespace identity must include all model/execution settings
/// affecting the saved state. The caller must provide a complete, immutable
/// value and restore request-private state before mutating it.
///
/// This type is deliberately not Sendable or internally synchronized. Own it
/// on the inference executor; Value itself need not conform to Sendable.
/// `maxBytes` bounds caller-reported logical payload bytes, and `maxKeyTokens`
/// independently bounds the total full-key token count. Neither represents a
/// physical-memory reservation. Entry count also bounds namespace/node count.
public final class QwenPrefixCacheIndex<Value> {
    public enum ConfigurationError: Error, Equatable {
        case invalidLimits
    }

    public let maxEntries: Int
    public let maxBytes: Int
    public let maxKeyTokens: Int

    private final class Entry {
        let namespace: String
        let tokens: [Int32]
        let value: Value
        let bytes: Int
        weak var previous: Entry?
        var next: Entry?

        init(namespace: String, tokens: [Int32], value: Value, bytes: Int) {
            self.namespace = namespace; self.tokens = tokens
            self.value = value; self.bytes = bytes
        }
    }

    private final class Node {
        var edge: [Int32]
        var children: [Int32: Node] = [:]
        var entry: Entry?

        init(edge: [Int32], entry: Entry? = nil) {
            self.edge = edge; self.entry = entry
        }
    }

    private var roots: [String: Node] = [:]
    private var oldest: Entry?
    private var newest: Entry?
    private var entryCount = 0
    private var payloadBytes = 0
    private var tokenCount = 0
    private var hits = 0
    private var misses = 0
    private var evictions = 0

    public init(maxEntries: Int, maxBytes: Int,
                maxKeyTokens: Int = 1_048_576) throws {
        guard maxEntries > 0, maxBytes > 0, maxKeyTokens > 0 else {
            throw ConfigurationError.invalidLimits
        }
        self.maxEntries = maxEntries; self.maxBytes = maxBytes
        self.maxKeyTokens = maxKeyTokens
    }

    public var statistics: QwenPrefixCacheStatistics {
        .init(hits: hits, misses: misses, evictions: evictions,
              entries: entryCount, logicalPayloadBytes: payloadBytes,
              keyTokens: tokenCount)
    }

    /// Inserts/replaces one complete snapshot and makes it most recently used.
    /// An empty key, negative byte cost, or individually oversized entry is
    /// rejected without changing entries, LRU order, or statistics. A valid
    /// replacement releases the old value and charges only the replacement.
    /// Capacity pressure evicts the least recently used entries globally,
    /// across namespaces. A replacement itself is not counted as an eviction.
    @discardableResult
    public func insert(tokens: [Int32], namespace: String, value: Value,
                       logicalPayloadBytes: Int) -> Bool {
        guard !tokens.isEmpty, logicalPayloadBytes >= 0,
              logicalPayloadBytes <= maxBytes, tokens.count <= maxKeyTokens else {
            return false
        }
        if let existing = exactEntry(tokens: tokens, namespace: namespace) {
            removeEntry(existing)
        }
        // Subtraction avoids overflow even when callers configure Int.max.
        while entryCount >= maxEntries || logicalPayloadBytes > maxBytes - payloadBytes ||
                tokens.count > maxKeyTokens - tokenCount {
            guard let victim = oldest else { preconditionFailure("prefix cache accounting mismatch") }
            removeEntry(victim)
            evictions += 1
        }
        let entry = Entry(namespace: namespace, tokens: tokens, value: value,
                          bytes: logicalPayloadBytes)
        insertIntoTree(entry)
        appendNewest(entry)
        entryCount += 1; payloadBytes += logicalPayloadBytes; tokenCount += tokens.count
        return true
    }

    /// Finds the longest inserted prefix no longer than maxPrefixTokens.
    /// A nonpositive cap is a miss. The complete cached key must match actual
    /// input tokens, so a match cannot cross a partially matched radix edge.
    /// Every call counts a hit/miss; touch=false preserves the LRU order.
    public func lookup(tokens: [Int32], namespace: String,
                       maxPrefixTokens: Int? = nil, touch: Bool = true) -> QwenPrefixCacheMatch<Value>? {
        guard let entry = longestEntry(tokens: tokens, namespace: namespace,
                                       maxPrefixTokens: maxPrefixTokens) else {
            misses += 1
            return nil
        }
        hits += 1
        if touch { makeNewest(entry) }
        return match(entry)
    }

    /// Inspects the same longest-prefix match without changing LRU or counters.
    public func peek(tokens: [Int32], namespace: String,
                     maxPrefixTokens: Int? = nil) -> QwenPrefixCacheMatch<Value>? {
        longestEntry(tokens: tokens, namespace: namespace,
                     maxPrefixTokens: maxPrefixTokens).map(match)
    }

    /// Removes only the exact key; an ancestor checkpoint is never removed by
    /// accident. Explicit removal is not a capacity eviction or lookup.
    @discardableResult
    public func remove(tokens: [Int32], namespace: String) -> Bool {
        guard let entry = exactEntry(tokens: tokens, namespace: namespace) else { return false }
        removeEntry(entry)
        return true
    }

    /// Releases every cache-owned value. Outstanding lookup results may still
    /// own values. Hit/miss/eviction history is retained unless explicitly reset.
    public func clear(resetStatistics: Bool = false) {
        // Break the LRU chain iteratively instead of retaining a long chain
        // through the newest/oldest roots while releasing the radix tree.
        while let entry = oldest { unlink(entry) }
        roots.removeAll(keepingCapacity: false)
        entryCount = 0; payloadBytes = 0; tokenCount = 0
        if resetStatistics { hits = 0; misses = 0; evictions = 0 }
    }

    private func match(_ entry: Entry) -> QwenPrefixCacheMatch<Value> {
        .init(prefixTokenCount: entry.tokens.count, value: entry.value,
              logicalPayloadBytes: entry.bytes)
    }

    private func longestEntry(tokens: [Int32], namespace: String,
                              maxPrefixTokens: Int?) -> Entry? {
        let limit = min(tokens.count, maxPrefixTokens ?? tokens.count)
        guard limit > 0, var node = roots[namespace] else { return nil }
        var offset = 0
        var best: Entry?
        while offset < limit, let child = node.children[tokens[offset]] {
            guard child.edge.count <= limit - offset else { break }
            var matched = true
            for i in child.edge.indices where child.edge[i] != tokens[offset + i] {
                matched = false
                break
            }
            guard matched else { break }
            offset += child.edge.count
            node = child
            if let entry = node.entry { best = entry }
        }
        return best
    }

    private func exactEntry(tokens: [Int32], namespace: String) -> Entry? {
        guard let entry = longestEntry(tokens: tokens, namespace: namespace,
                                       maxPrefixTokens: nil),
              entry.tokens.count == tokens.count else { return nil }
        return entry
    }

    private func insertIntoTree(_ entry: Entry) {
        let root: Node
        if let existing = roots[entry.namespace] { root = existing }
        else {
            root = Node(edge: [])
            roots[entry.namespace] = root
        }
        let tokens = entry.tokens
        var node = root
        var offset = 0
        while offset < tokens.count {
            let first = tokens[offset]
            guard let child = node.children[first] else {
                node.children[first] = Node(edge: Array(tokens[offset...]), entry: entry)
                return
            }
            var common = 0
            while common < child.edge.count && common < tokens.count - offset &&
                    child.edge[common] == tokens[offset + common] {
                common += 1
            }
            if common == child.edge.count {
                node = child; offset += common
                continue
            }
            // The first token is the dictionary key, so common is at least 1.
            let split = Node(edge: Array(child.edge.prefix(common)))
            child.edge = Array(child.edge.dropFirst(common))
            split.children[child.edge[0]] = child
            node.children[first] = split
            offset += common
            if offset == tokens.count { split.entry = entry }
            else {
                split.children[tokens[offset]] = Node(edge: Array(tokens[offset...]), entry: entry)
            }
            return
        }
        node.entry = entry
    }

    private func removeEntry(_ entry: Entry) {
        guard let root = roots[entry.namespace] else {
            preconditionFailure("prefix cache namespace missing")
        }
        var path: [(parent: Node, key: Int32, child: Node)] = []
        var node = root
        var offset = 0
        while offset < entry.tokens.count {
            let key = entry.tokens[offset]
            guard let child = node.children[key] else {
                preconditionFailure("prefix cache entry missing")
            }
            path.append((node, key, child))
            offset += child.edge.count
            node = child
        }
        precondition(node.entry === entry)
        node.entry = nil
        for step in path.reversed() where step.child.entry == nil {
            if step.child.children.isEmpty {
                step.parent.children.removeValue(forKey: step.key)
            } else if step.child.children.count == 1, let child = step.child.children.values.first {
                step.child.edge += child.edge
                step.child.entry = child.entry
                step.child.children = child.children
            }
        }
        if root.children.isEmpty { roots.removeValue(forKey: entry.namespace) }
        unlink(entry)
        entryCount -= 1; payloadBytes -= entry.bytes; tokenCount -= entry.tokens.count
    }

    private func appendNewest(_ entry: Entry) {
        entry.previous = newest
        newest?.next = entry
        newest = entry
        if oldest == nil { oldest = entry }
    }

    private func unlink(_ entry: Entry) {
        let previous = entry.previous
        let next = entry.next
        if let previous { previous.next = next } else { oldest = next }
        if let next { next.previous = previous } else { newest = previous }
        entry.previous = nil; entry.next = nil
    }

    private func makeNewest(_ entry: Entry) {
        guard newest !== entry else { return }
        unlink(entry)
        appendNewest(entry)
    }
}
