import ANERunnerCore
import Foundation

/// An executor-confined, single-forward authorization for optional capacity
/// append. Only the model can create one after checking its joint byte budget.
/// There is no public initializer or release operation. No tensor, state or
/// model is retained, so this owner does not itself inhibit MLX donation.
public final class QwenKVCapacityAppendPermit {
    let workspaceBytes: Int
    private let modelOwner, sessionIdentity: UUID
    private let offset, rowLimit: Int
    private var lease: QwenStateBudget.Lease?
    private var consumed = false
    private var released = false

    init(modelOwner: UUID, sessionIdentity: UUID, offset: Int, rowLimit: Int,
         workspaceBytes: Int, lease: QwenStateBudget.Lease?) {
        self.modelOwner = modelOwner; self.sessionIdentity = sessionIdentity
        self.offset = offset; self.rowLimit = rowLimit
        self.workspaceBytes = workspaceBytes; self.lease = lease
    }

    /// Pure host validation; a failed owner/position check does not consume the
    /// permit. Successful use authorizes exactly one explicit AR forward.
    func consume(modelOwner: UUID, sessionIdentity: UUID, offset: Int,
                 maximumRowLimit: Int) throws -> Int {
        guard !consumed, !released, self.modelOwner == modelOwner,
              self.sessionIdentity == sessionIdentity, self.offset == offset,
              offset >= 0, rowLimit > offset, rowLimit <= maximumRowLimit,
              workspaceBytes >= 0,
              (workspaceBytes == 0 ? lease == nil :
                (lease?.kind == .workspace && lease?.bytes == workspaceBytes && lease?.isReleased == false)) else {
            throw GPUError.invalid("KV capacity permit is invalid, consumed or belongs to another model/state/offset")
        }
        consumed = true
        return rowLimit
    }

    /// The decode step calls this only after its selected/state synchronous
    /// join, or after the outer error handler attempts device recovery. Failed
    /// recovery disables the model; zero logical bytes do not prove device idle.
    func releaseAfterCompletion() {
        guard !released else { return }
        released = true
        lease?.release(); lease = nil
    }

    deinit { lease?.release() }
}
