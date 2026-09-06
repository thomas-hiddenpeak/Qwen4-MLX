// PLE math follows garnermccloud/mlx-serve transformer.zig,
// commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1 (MIT; see UPSTREAM-LICENSE).
import ANERunnerCore
import CMLX
import Dispatch
import Foundation
import Synchronization

/// CPU-only read job. The Sendable result contains no MLX tensor or model state.
public final class PLEReadTask: Sendable {
    private let group = DispatchGroup()
    private let result = Mutex<Result<[Float], any Error>?>(nil)
    public let logicalBytes: Int
    init(table: NGramTable, rows: [Int], workers: Int, queue: DispatchQueue? = nil) {
        logicalBytes = rows.count * table.dimension
        group.enter()
        (queue ?? DispatchQueue.global(qos: .userInitiated)).async { [self] in
            let value = Result { try GPUSSDReader.readRows(table: table, rows: rows, workers: workers) }
            result.withLock { $0 = value }
            group.leave()
        }
    }
    /// Join submitted work on cancellation/error before starting another request.
    /// Call from the request owner, never from the queue executing this task.
    /// A running pread is not interrupted; the owner's lookahead bounds the work.
    public func drain() { group.wait() }
    public func wait() throws -> [Float] {
        drain()
        guard let value = result.withLock({ $0 }) else { throw GPUError.invalid("Missing PLE read result") }
        return try value.get()
    }
}

public final class GPUPLE {
    /// CPU-only lookahead. Creating this value never mutates model state or
    /// carries an MLX object into the read queue. The caller bounds live chunks.
    public struct PreparedInput: Sendable {
        public let tokens: [Int32]
        public let historyBefore: [UInt32]
        public let historyAfter: [UInt32]
        public let readTask: PLEReadTask
        private let initialHistory: [UInt32]
        private let table: NGramTable

        // Shared by the model wrapper and small CPU-only tests.
        init(tokens: [Int32], history: [UInt32], hash: NGramHash, table: NGramTable,
             workers: Int, queue: DispatchQueue? = nil) throws {
            guard (1...GPUSSDReader.maximumWorkers).contains(workers) else {
                throw GPUError.invalid("Invalid SSD worker count")
            }
            guard tokens.allSatisfy({ $0 >= 0 }) else {
                throw GPUError.invalid("PLE prefetch token IDs must be nonnegative")
            }
            let before = history.isEmpty ? hash.initialHistory : history
            let ids = tokens.map(UInt32.init)
            let rows = try hash.rowIDs(previousTokens: before, tokens: ids)
            let after = try hash.history(after: ids, previousTokens: before)
            self.tokens = tokens
            historyBefore = before
            historyAfter = after
            initialHistory = hash.initialHistory
            self.table = table
            readTask = PLEReadTask(table: table, rows: rows, workers: workers, queue: queue)
        }

        func consume(tokens: [Int32], history: inout [UInt32], table: NGramTable) throws -> PLEReadTask {
            guard table === self.table else { throw GPUError.invalid("Prepared PLE input belongs to another table") }
            guard tokens == self.tokens else { throw GPUError.invalid("Prepared PLE token IDs do not match") }
            let before = history.isEmpty ? initialHistory : history
            guard before == historyBefore else { throw GPUError.invalid("Prepared PLE history does not match") }
            history = historyAfter
            return readTask
        }
    }

    /// Optional data for committing an accepted prefix of one verification
    /// forward. Token/hash history is supplied from the caller's checkpoint,
    /// because prepare/consume advance it before forward starts.
    public struct VerificationCapture {
        public let convInputs: Tensor
        public let tokenCount: Int
        fileprivate let owner: ObjectIdentifier
        fileprivate let hadPreviousConvolution: Bool
    }

    public struct State {
        public var history: [UInt32] = []
        public var convolution: Tensor?
        public fileprivate(set) var verificationCapture: VerificationCapture?
        public init() {}
        public mutating func reset() { self = State() }
        public mutating func clearVerificationCapture() { verificationCapture = nil }
        public var tensors: [Tensor] {
            [convolution, verificationCapture?.convInputs].compactMap { $0 }
        }
    }
    private let key, value, normKey, normQuery, normConv, conv: Tensor
    private let hash: NGramHash
    public let table: NGramTable
    private let hidden, streams, dilation, stateLength: Int
    private let epsilon: Float
    private let workers: Int

    public init(weights: GPUWeights, configuration c: QwenConfiguration, layer: Int, ordinal: Int, workers: Int = 1) throws {
        guard (1...GPUSSDReader.maximumWorkers).contains(workers) else { throw GPUError.invalid("Invalid SSD worker count") }
        self.workers = workers
        hidden = c.hiddenSize; streams = c.hcCount; dilation = c.ngramSize
        stateLength = (c.pleConvKernel - 1) * dilation; epsilon = Float(c.rmsNormEpsilon)
        let p = "language_model.model.layers.\(layer).ple"
        key = try MX.transpose(weights.tensor(p + ".key_proj.weight"), [1, 0])
        value = try MX.transpose(weights.tensor(p + ".value_proj.weight"), [1, 0])
        normKey = try MX.reshape(weights.tensor(p + ".norm_key.weight"), [streams, hidden])
        normQuery = try MX.reshape(weights.tensor(p + ".norm_query.weight"), [streams, hidden])
        normConv = try MX.reshape(weights.tensor(p + ".norm_conv.weight"), [streams, hidden])
        conv = try weights.tensor(p + ".conv1d.weight")
        guard key.shape == [c.pleEmbeddingDimension, streams * hidden], value.shape == [c.pleEmbeddingDimension, hidden],
              conv.shape == [streams * hidden, c.pleConvKernel, 1] else { throw GPUError.invalid("Unsupported PLE shape") }
        hash = try NGramHash(unigramVocabularySize: UInt32(c.vocabularySize), ngramSize: c.ngramSize,
            headsPerNGram: c.ngramHeadsPerOrder, vocabularyBase: UInt64(c.ngramVocabularyBase),
            vocabularyDivisor: UInt64(c.ngramDivisor), pleLayerIndex: UInt32(ordinal), eosTokenID: UInt32(c.eosTokenID))
        guard !c.ngramTableFile.contains(".."), !c.ngramTableFile.hasPrefix("/") else { throw GPUError.invalid("Invalid PLE table path") }
        table = try NGramTable(url: c.modelDirectory.appendingPathComponent(c.ngramTableFile))
        guard table.rowCount == hash.totalRows, table.dimension * hash.headCount == c.pleEmbeddingDimension,
              table.scale == Float(c.ngramScale) else { throw GPUError.invalid("PLE table/configuration mismatch") }
    }
    public func prepare(tokens: [Int32], state: inout State) throws -> PLEReadTask {
        let history = state.history.isEmpty ? hash.initialHistory : state.history
        let ids = tokens.map(UInt32.init)
        let rows = try hash.rowIDs(previousTokens: history, tokens: ids)
        state.history = try hash.history(after: ids, previousTokens: history)
        return PLEReadTask(table: table, rows: rows, workers: workers)
    }
    /// A serial request-owned queue preserves chunk submission order while each
    /// read still uses the configured SSD worker count internally.
    public func prefetch(tokens: [Int32], history: [UInt32], queue: DispatchQueue? = nil) throws -> PreparedInput {
        try PreparedInput(tokens: tokens, history: history, hash: hash, table: table, workers: workers, queue: queue)
    }
    public func consume(prepared: PreparedInput, tokens: [Int32], state: inout State) throws -> PLEReadTask {
        try prepared.consume(tokens: tokens, history: &state.history, table: table)
    }
    public func forward(_ stream: Tensor, embedding: Tensor, state: inout State,
                        captureVerification: Bool = false,
                        verificationLinear: GPUVerificationLinear? = nil) throws -> Tensor {
        let n = stream.shape[1], width = streams * hidden
        func norm(_ x: Tensor, _ w: Tensor) throws -> Tensor {
            try GPUHyperConnection.groupNorm(x, weight: w, streams: streams, hidden: hidden, epsilon: epsilon)
        }
        let k = try norm(MX.linear(embedding, key, verification: verificationLinear), normKey)
        let v = try MX.linear(embedding, value, verification: verificationLinear)
        let q = try norm(stream, normQuery)
        let gate = try MX.mul(MX.sum(MX.mul(k, q), axis: -1, keepDims: true), MX.scalar(1 / sqrt(Float(hidden)), MLX_BFLOAT16))
        let signedRoot = try MX.mul(MX.sqrt(MX.maximum(MX.abs(gate), MX.scalar(1e-6, MLX_BFLOAT16))), MX.sign(gate))
        let gv = try MX.reshape(MX.mul(MX.sigmoid(signedRoot), MX.reshape(v, [1, n, 1, hidden])), [1, n, width])
        let normalized = try MX.reshape(norm(gv, normConv), [1, n, width])
        let previous = try state.convolution ?? MX.zeros([1, stateLength, width], MLX_BFLOAT16)
        let cat = try MX.concat([previous, normalized], axis: 1)
        let convolution = try MX.silu(MX.conv1d(cat, weight: conv, dilation: dilation, groups: width))
        let nextConvolution = try MX.contiguous(MX.copy(MX.slice(cat, starts: [0, n, 0], ends: [1, n + stateLength, width])))
        let output = try MX.add(gv, convolution)
        state.verificationCapture = captureVerification
            ? VerificationCapture(convInputs: cat, tokenCount: n, owner: ObjectIdentifier(self),
                                  hadPreviousConvolution: state.convolution != nil) : nil
        state.convolution = nextConvolution
        return output
    }

    /// Commit `count` input rows from the captured forward, without recomputing
    /// projections or convolution. `tokens` is the entire verification input;
    /// `previousHistory` is the matching pre-prepare checkpoint's hash history.
    /// The returned tensors are lazy independent copies: jointly evaluate them
    /// before dropping the verification state and publishing this new state.
    public func committingPrefix(_ state: State, count: Int, tokens: [Int32],
                                 previousHistory: [UInt32]) throws -> State {
        guard let capture = state.verificationCapture,
              capture.owner == ObjectIdentifier(self),
              count >= 0, count <= capture.tokenCount,
              tokens.count == capture.tokenCount, tokens.allSatisfy({ $0 >= 0 }),
              capture.convInputs.shape == [1, stateLength + capture.tokenCount, streams * hidden],
              capture.convInputs.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("PLE prefix requires this layer's captured verification input and a valid prefix count")
        }
        let before = previousHistory.isEmpty ? hash.initialHistory : previousHistory
        let ids = tokens.map(UInt32.init)
        guard try hash.history(after: ids, previousTokens: before) == state.history else {
            throw GPUError.invalid("PLE verification tokens/history do not match the completed lookup state")
        }
        if count == capture.tokenCount {
            var complete = state
            complete.clearVerificationCapture()
            return complete
        }
        var prefix = State()
        prefix.history = count == 0 ? previousHistory
            : try hash.history(after: Array(ids.prefix(count)), previousTokens: before)
        if count > 0 || capture.hadPreviousConvolution {
            let selected = try MX.slice(capture.convInputs, starts: [0, count, 0],
                                        ends: [1, count + stateLength, streams * hidden])
            prefix.convolution = try GPUVerificationCopy.tensor(selected)
        }
        return prefix
    }
}
