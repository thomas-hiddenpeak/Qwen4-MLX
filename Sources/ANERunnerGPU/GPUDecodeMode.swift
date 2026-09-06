/// Model-specific candidates. The reference remains available for same-process
/// comparisons; these switches never change quantization or enable MTP.
public enum GPUDecodeMode: String, Codable, CaseIterable, Sendable {
    case reference, scalar, elementwise, projections, all

    public var usesNativeTokenReadback: Bool { self != .reference }
    public var fusesSharedElementwise: Bool { self == .elementwise || self == .all }
    public var fusesProjections: Bool { self == .projections || self == .all }
}
