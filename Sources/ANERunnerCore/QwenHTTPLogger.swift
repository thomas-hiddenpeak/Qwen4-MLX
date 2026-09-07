import Darwin
import Foundation

/// One dedicated writer for the experimental HTTP process. Producers hold only
/// a short queue lock; no write or wait for the sink occurs under that lock.
/// stop() drops queued records and never joins a potentially blocked writer.
public final class QwenHTTPLogger: @unchecked Sendable {
    public struct Limits: Sendable {
        public let maxBytes, maxEvents, maxEventBytes: Int
        public init(maxBytes: Int = 65_536, maxEvents: Int = 128, maxEventBytes: Int = 4096) {
            self.maxBytes = maxBytes; self.maxEvents = maxEvents; self.maxEventBytes = maxEventBytes
        }
    }

    public enum ConfigurationError: Error { case invalidLimits }
    public enum SignalPolicyError: Error {
        case requiresIgnoredSIGPIPE
        case queryFailed(Int32)
    }

    public struct Snapshot: Sendable {
        public let accepting, writerExited: Bool
        public let maxBytes, maxEvents, maxEventBytes: Int
        public let bufferedBytes, bufferedEvents, queuedEvents, inFlightBytes: Int
        public let enqueuedEvents, writtenEvents, droppedEvents, droppedBytes, writeFailures: UInt64
        public let lastWriteErrno: Int32?
    }

    private let limits: Limits
    private let condition = NSCondition()
    private let sink: @Sendable (Data) -> Int32?
    private var queue: [Data] = []
    private var accepting = true, writerExited = false
    private var bufferedBytes = 0, inFlightBytes = 0
    private var enqueuedEvents: UInt64 = 0, writtenEvents: UInt64 = 0
    private var droppedEvents: UInt64 = 0, droppedBytes: UInt64 = 0, writeFailures: UInt64 = 0
    private var lastWriteErrno: Int32?

    /// stderr may be a pipe. Darwin reports its EPIPE signal to the process,
    /// so the owning CLI must explicitly keep SIGPIPE ignored for this logger's
    /// lifetime. Core only verifies the policy; it never changes it implicitly.
    public convenience init(limits: Limits = .init()) throws {
        try self.init(limits: limits, fileDescriptor: STDERR_FILENO)
    }

    /// Same production write path on an owned CPU-test descriptor. The caller
    /// owns the fd and must keep it open until the writer has exited.
    convenience init(limits: Limits, fileDescriptor: Int32) throws {
        var action = sigaction()
        guard sigaction(SIGPIPE, nil, &action) == 0 else {
            throw SignalPolicyError.queryFailed(errno)
        }
        let handler = unsafeBitCast(action.__sigaction_u.__sa_handler, to: UnsafeRawPointer?.self)
        let ignored = unsafeBitCast(SIG_IGN, to: UnsafeRawPointer?.self)
        guard handler == ignored else { throw SignalPolicyError.requiresIgnoredSIGPIPE }
        try self.init(limits: limits, sink: { data in
            data.withUnsafeBytes { bytes -> Int32? in
                var offset = 0
                while offset < bytes.count {
                    let count = Darwin.write(fileDescriptor, bytes.baseAddress!.advanced(by: offset), bytes.count - offset)
                    if count > 0 { offset += count }
                    else if count < 0 && errno == EINTR { continue }
                    else { return count == 0 ? EIO : errno }
                }
                return nil
            }
        })
    }

    /// The sink seam is internal and used only by CPU boundary tests. It is
    /// always invoked on this logger's one fixed thread, never on a producer.
    init(limits: Limits, sink: @escaping @Sendable (Data) -> Int32?) throws {
        guard limits.maxBytes > 0, limits.maxEvents > 0, limits.maxEventBytes > 0,
              limits.maxEventBytes <= limits.maxBytes else { throw ConfigurationError.invalidLimits }
        self.limits = limits; self.sink = sink
        let thread = Thread { [self] in writerLoop() }
        thread.name = "ane-runner.http.log-writer"
        thread.start()
    }

    /// Data includes its newline. An oversized record is dropped, never split
    /// into invalid JSON. Accounting includes the record currently in write().
    @discardableResult
    public func enqueue(_ data: Data) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard accepting, !data.isEmpty, data.count <= limits.maxEventBytes,
              data.count <= limits.maxBytes - bufferedBytes,
              queue.count + (inFlightBytes > 0 ? 1 : 0) < limits.maxEvents else {
            drop(events: 1, bytes: data.count); return false
        }
        queue.append(data); bufferedBytes += data.count
        enqueuedEvents = adding(enqueuedEvents, 1)
        condition.signal(); return true
    }

    public func snapshot() -> Snapshot {
        condition.lock(); defer { condition.unlock() }
        return Snapshot(accepting: accepting, writerExited: writerExited,
            maxBytes: limits.maxBytes, maxEvents: limits.maxEvents, maxEventBytes: limits.maxEventBytes,
            bufferedBytes: bufferedBytes, bufferedEvents: queue.count + (inFlightBytes > 0 ? 1 : 0),
            queuedEvents: queue.count, inFlightBytes: inFlightBytes,
            enqueuedEvents: enqueuedEvents, writtenEvents: writtenEvents, droppedEvents: droppedEvents,
            droppedBytes: droppedBytes, writeFailures: writeFailures, lastWriteErrno: lastWriteErrno)
    }

    /// Nonblocking with respect to sink IO. The single in-flight Data remains
    /// charged until write returns or this CLI process exits; it is never freed
    /// early, interrupted by closing shared stderr, or waited on at shutdown.
    public func stop() {
        condition.lock(); defer { condition.unlock() }
        accepting = false
        discardQueued()
        condition.broadcast()
    }

    private func writerLoop() {
        while true {
            condition.lock()
            while queue.isEmpty && accepting { condition.wait() }
            guard !queue.isEmpty else {
                writerExited = true
                condition.unlock(); return
            }
            let data = queue.removeFirst()
            inFlightBytes = data.count
            condition.unlock()

            let failure = autoreleasepool { sink(data) }

            condition.lock()
            bufferedBytes -= data.count; inFlightBytes = 0
            if let failure {
                drop(events: 1, bytes: data.count)
                recordFailure(failure)
            } else {
                writtenEvents = adding(writtenEvents, 1)
            }
            condition.unlock()
        }
    }

    // The following helpers are called only with condition held. UInt64
    // counters saturate rather than wrap during an indefinitely running server.
    private func adding(_ a: UInt64, _ b: UInt64) -> UInt64 {
        let (sum, overflow) = a.addingReportingOverflow(b)
        return overflow ? .max : sum
    }
    private func drop(events: Int, bytes: Int) {
        droppedEvents = adding(droppedEvents, UInt64(events))
        droppedBytes = adding(droppedBytes, UInt64(bytes))
    }
    private func discardQueued() {
        drop(events: queue.count, bytes: bufferedBytes - inFlightBytes)
        queue.removeAll(keepingCapacity: false)
        bufferedBytes = inFlightBytes
    }
    private func recordFailure(_ code: Int32) {
        writeFailures = adding(writeFailures, 1); lastWriteErrno = code
        accepting = false; discardQueued()
        condition.broadcast()
    }
}
