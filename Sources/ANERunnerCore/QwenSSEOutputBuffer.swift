import Foundation
import Synchronization

/// One request's encoded SSE body frames, shared by the inference producer and
/// the network writer. Enqueue never waits for socket progress; only short state
/// operations hold the mutex. No user callback or network operation runs locked.
///
/// The caller supplies complete UTF-8 SSE frames and acts on returned Actions.
/// Exactly one send may be outstanding. Its bytes/events remain charged until
/// acknowledgeSend, including after disconnect, because the transport can still
/// retain that Data. The limits cover SSE body bytes, not HTTP/OS send buffers.
public final class QwenSSEOutputBuffer: Sendable {
    public struct Limits: Sendable {
        public let maxBytes, maxEvents, terminalReserveBytes: Int
        public init(maxBytes: Int = 65_536, maxEvents: Int = 256,
                    terminalReserveBytes: Int = 4_096) {
            self.maxBytes = maxBytes; self.maxEvents = maxEvents
            self.terminalReserveBytes = terminalReserveBytes
        }
    }
    public enum ConfigurationError: Error, Equatable {
        case invalidLimits, invalidOverflowFrame
    }
    public enum Outcome: String, Sendable {
        case completed, failed, cancelled, slowConsumer, disconnected
    }
    /// Only a scheduler terminal event calls finish. A transport failure may
    /// already have selected a different outcome, which finish must preserve.
    public enum Completion: Sendable { case completed, failed, cancelled }
    public struct Actions: Equatable, Sendable {
        /// Coalesced wakeup for an idle writer, or the next send after an ack.
        public let scheduleSend: Bool
        /// Call the request's cancellation token outside this object's mutex.
        /// Emitted at most once, for overflow/disconnect/send failure.
        public let cancelProducer: Bool
        public static let none = Actions(scheduleSend: false, cancelProducer: false)
    }
    public enum EnqueueStatus: Equatable, Sendable { case accepted, overflow, closed }
    public struct EnqueueResult: Sendable {
        public let status: EnqueueStatus
        public let actions: Actions
    }
    public enum FinishStatus: Equatable, Sendable { case accepted, alreadyTerminal, invalidFrame }
    public struct FinishResult: Sendable {
        public let status: FinishStatus
        public let actions: Actions
    }
    public struct Send: Sendable {
        public let id: UInt64
        public let data: Data
        public let isTerminal: Bool
    }
    public struct Snapshot: Sendable {
        public let bufferedBytes, bufferedEvents, queuedEvents: Int
        public let inFlightBytes: Int
        public let hasInFlight: Bool
        public let outcome: Outcome?
        public let producerFinished, transportClosed, cancellationRequested: Bool
        /// All accepted frames have been processed or explicitly discarded.
        /// This does not mean the remote client received them.
        public let isDrained: Bool
    }

    private struct State: Sendable {
        var queue: [Data] = []
        var terminalFrame: Data?
        var inFlight: Send?
        var bytes = 0, events = 0
        var nextSendID: UInt64 = 1
        var outcome: Outcome?
        var producerFinished = false, transportClosed = false, cancellationRequested = false
    }
    public let limits: Limits
    private let overflowFrame: Data
    private let state = Mutex(State())

    /// Reserve one event and terminalReserveBytes inside the total limits.
    /// overflowFrame should be a small error SSE frame (and optional [DONE]);
    /// it is retained separately so a full token queue can still end explicitly.
    public init(limits: Limits = Limits(), overflowFrame: Data) throws {
        guard limits.maxEvents >= 2, limits.terminalReserveBytes > 0,
              limits.maxBytes > limits.terminalReserveBytes else {
            throw ConfigurationError.invalidLimits
        }
        guard !overflowFrame.isEmpty, overflowFrame.count <= limits.terminalReserveBytes else {
            throw ConfigurationError.invalidOverflowFrame
        }
        self.limits = limits; self.overflowFrame = overflowFrame
    }

    /// Called synchronously from onToken. On overflow the offered frame is not
    /// accepted, previously accepted frames stay ordered, and the reserved
    /// terminal frame follows them. Cancel/throw out of the generation callback;
    /// never retry that token or wait here for the client to catch up.
    public func enqueue(_ frame: Data) -> EnqueueResult {
        state.withLock { state in
            guard state.outcome == nil && !state.transportClosed else {
                return EnqueueResult(status: .closed, actions: .none)
            }
            let wake = state.inFlight == nil && state.events == 0
            let regularByteLimit = limits.maxBytes - limits.terminalReserveBytes
            if frame.count > regularByteLimit - state.bytes || state.events >= limits.maxEvents - 1 {
                state.outcome = .slowConsumer
                state.terminalFrame = overflowFrame
                state.bytes += overflowFrame.count; state.events += 1
                let cancel = Self.requestCancellation(&state)
                return EnqueueResult(status: .overflow,
                    actions: Actions(scheduleSend: wake, cancelProducer: cancel))
            }
            state.queue.append(frame)
            state.bytes += frame.count; state.events += 1
            return EnqueueResult(status: .accepted,
                actions: Actions(scheduleSend: wake, cancelProducer: false))
        }
    }

    /// First terminal selection wins across completion, overflow and disconnect.
    /// Invalid terminal Data leaves the request open so the adapter can provide
    /// a bounded fallback. A late scheduler event records producer completion
    /// even when an earlier transport outcome has already won.
    public func finish(_ completion: Completion, frame: Data) -> FinishResult {
        state.withLock { state in
            if state.outcome != nil {
                state.producerFinished = true
                return FinishResult(status: .alreadyTerminal, actions: .none)
            }
            guard !frame.isEmpty && frame.count <= limits.terminalReserveBytes else {
                return FinishResult(status: .invalidFrame, actions: .none)
            }
            let wake = state.inFlight == nil && state.events == 0
            switch completion {
            case .completed: state.outcome = .completed
            case .failed: state.outcome = .failed
            case .cancelled: state.outcome = .cancelled
            }
            state.producerFinished = true
            state.terminalFrame = frame
            state.bytes += frame.count; state.events += 1
            return FinishResult(status: .accepted,
                actions: Actions(scheduleSend: wake, cancelProducer: false))
        }
    }

    /// Writer only: acquire the next frame. A lease is never issued twice.
    /// Retain the returned id until Network's contentProcessed callback and
    /// call acknowledgeSend exactly for that id, including on send failure.
    public func beginSend() -> Send? {
        state.withLock { state in
            guard !state.transportClosed && state.inFlight == nil else { return nil }
            let data: Data, terminal: Bool
            if !state.queue.isEmpty {
                data = state.queue.removeFirst(); terminal = false
            } else if let frame = state.terminalFrame {
                data = frame; terminal = true; state.terminalFrame = nil
            } else { return nil }
            let send = Send(id: state.nextSendID, data: data, isTerminal: terminal)
            state.nextSendID += 1
            state.inFlight = send
            return send
        }
    }

    /// A matching send callback releases its quota. Duplicate/stale callbacks
    /// are harmless and cannot cancel a newer send. Success means the transport
    /// processed this content; it is not a remote delivery acknowledgement.
    public func acknowledgeSend(_ id: UInt64, succeeded: Bool) -> Actions {
        state.withLock { state in
            guard let send = state.inFlight, send.id == id else { return .none }
            state.inFlight = nil
            state.bytes -= send.data.count; state.events -= 1
            if !succeeded { return Self.closeTransport(&state) }
            return Actions(scheduleSend: !state.transportClosed && state.events > 0, cancelProducer: false)
        }
    }

    /// The adapter also cancels/closes NWConnection and stops its receive loop.
    /// Drop unsent frames immediately; keep any in-flight quota until its send
    /// callback. A completed inference outcome is never rewritten as a second
    /// terminal event, and no terminal frame is promised to a disconnected peer.
    public func disconnect() -> Actions { state.withLock { Self.closeTransport(&$0) } }

    public func snapshot() -> Snapshot {
        state.withLock { state in
            Snapshot(bufferedBytes: state.bytes, bufferedEvents: state.events,
                queuedEvents: state.queue.count + (state.terminalFrame == nil ? 0 : 1),
                inFlightBytes: state.inFlight?.data.count ?? 0, hasInFlight: state.inFlight != nil,
                outcome: state.outcome, producerFinished: state.producerFinished,
                transportClosed: state.transportClosed, cancellationRequested: state.cancellationRequested,
                isDrained: state.outcome != nil && state.events == 0)
        }
    }

    private static func requestCancellation(_ state: inout State) -> Bool {
        guard !state.producerFinished && !state.cancellationRequested else { return false }
        state.cancellationRequested = true
        return true
    }
    private static func closeTransport(_ state: inout State) -> Actions {
        if state.outcome == nil { state.outcome = .disconnected }
        state.transportClosed = true
        state.queue.removeAll(keepingCapacity: false); state.terminalFrame = nil
        state.bytes = state.inFlight?.data.count ?? 0
        state.events = state.inFlight == nil ? 0 : 1
        return Actions(scheduleSend: false, cancelProducer: requestCancellation(&state))
    }
}
