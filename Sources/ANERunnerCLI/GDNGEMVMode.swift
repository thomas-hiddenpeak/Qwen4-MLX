import Darwin

/// Trial-local selection in the separately staged MLX experimental library.
enum GDNGEMVMode: String, CaseIterable {
    case reference, bm4, rows4, bm2, bm1, gemm, gemmSplit

    func apply() throws {
        guard let library = dlopen(nil, RTLD_NOW) else {
            throw CLIError.usage("Cannot inspect loaded MLX library")
        }
        defer { dlclose(library) }
        guard let symbol = dlsym(library, "anemlx_set_gdn_gemv_mode") else {
            if self == .reference { return }
            throw CLIError.usage("GDN GEMV experiments require the separately staged tuning library")
        }
        typealias Setter = @convention(c) (Int32) -> Void
        let setMode = unsafeBitCast(symbol, to: Setter.self)
        let value: Int32
        switch self {
        case .reference: value = 0
        case .bm4: value = 1
        case .rows4: value = 2
        case .bm2: value = 3
        case .bm1: value = 4
        case .gemm: value = 5
        case .gemmSplit: value = 6
        }
        typealias Maximum = @convention(c) () -> Int32
        let maximum = dlsym(library, "anemlx_gdn_gemv_max_mode").map {
            unsafeBitCast($0, to: Maximum.self)()
        } ?? 2
        guard value <= maximum else {
            throw CLIError.usage("This GDN tuning library does not support \(rawValue)")
        }
        setMode(value)
    }
}
