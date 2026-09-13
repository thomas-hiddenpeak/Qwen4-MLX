import Foundation
import XCTest
import ANERunnerCore
@testable import ANERunnerGPU

/// Error values only: no MX call, Tensor, model, native pool or device work.
final class GPUPrefixArchiveInvalidationTests: XCTestCase {
    func testDeviceAndRequestErrorsNeverEstablishDiskCorruption() {
        let errors: [Error] = [
            GPUError.invalid("MLX eval: [METAL] Command buffer execution failed"),
            GPUError.invalid("Could not create MLX array"),
            GPUError.invalid("MLX allocation: [malloc] Unable to allocate bytes"),
            QwenGenerationError.resourceLimit("optional cache budget"),
            QwenGenerationError.cancelled,
            QwenGenerationError.unavailable("failed device recovery"),
        ]
        for error in errors {
            XCTAssertFalse(QwenPrefixStateArchiveDescriptor.isInvalidArchive(error))
        }
    }
}
