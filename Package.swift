// swift-tools-version: 6.0

import PackageDescription
import Foundation

// Reuse the already pinned native MLX build; no model or Python subprocess is
// embedded in the runner. Override for a separately staged compatible build.
let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let mlxRoot = ProcessInfo.processInfo.environment["ANERUNNER_MLX_ROOT"] ??
    packageRoot.appendingPathComponent("../qwen38-ssd/runtime/mlx-serve/lib/mlx").standardizedFileURL.path
let mlxIncludes: [SwiftSetting] = [.unsafeFlags(["-Xcc", "-I\(mlxRoot)/include"])]
let mlxLinks: [LinkerSetting] = [.unsafeFlags(["-L\(mlxRoot)/lib", "-Xlinker", "-rpath", "-Xlinker", "\(mlxRoot)/lib"])]

let package = Package(
    name: "ANEModelRunner",
    platforms: [.macOS("26.2")],
    products: [
        .library(name: "ANERunnerCore", targets: ["ANERunnerCore"]),
        .library(name: "ANERunnerGPU", targets: ["ANERunnerGPU"]),
        .executable(name: "ane-runner", targets: ["ANERunnerCLI"]),
        .executable(name: "ane-telemetry", targets: ["ANERunnerTelemetry"]),
    ],
    targets: [
        .target(name: "ANERunnerCore"),
        .executableTarget(name: "ANERunnerTelemetry",
            cSettings: [.unsafeFlags(["-fobjc-arc", "-fblocks"])],
            linkerSettings: [.linkedFramework("Foundation"), .linkedFramework("IOKit"), .linkedLibrary("IOReport")]),
        .systemLibrary(name: "CMLX"),
        .target(name: "ANERunnerGPU", dependencies: ["CMLX", "ANERunnerCore"],
                swiftSettings: mlxIncludes, linkerSettings: mlxLinks),
        .executableTarget(name: "ANERunnerCLI", dependencies: ["ANERunnerCore", "ANERunnerGPU"],
                         swiftSettings: mlxIncludes, linkerSettings: mlxLinks),
        .testTarget(name: "ANERunnerCoreTests", dependencies: ["ANERunnerCore"]),
        .testTarget(name: "ANERunnerGPUTests", dependencies: ["ANERunnerGPU", "ANERunnerCore"],
                    swiftSettings: mlxIncludes, linkerSettings: mlxLinks),
    ]
)
