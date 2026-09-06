import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Bounded real-weight transaction/stop gate. One loaded trunk and head;
    /// no timing result is interpreted as a throughput benchmark.
    static func probeGPUMTPState(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--verification"])
        guard let verification = QwenMTPDecoder.Verification(rawValue: args["--verification"] ?? "scalar") else {
            throw CLIError.usage("Unknown MTP verification policy")
        }
        let cancellationVerification: QwenMTPDecoder.Verification = verification == .batchedTokenMoE ? verification : .batchedScalarMoE
        let historyCancellationVerification: QwenMTPDecoder.Verification = verification == .batchedTokenMoE ? verification : .batchedScalarLinear
        let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("MTP state probe --output must be a new file")
        }
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        let tokens: [Int32], inputSource: [String: Any]
        if let path = args["--tokens-file"] {
            let url = URL(fileURLWithPath: path).standardizedFileURL
            let data = try Data(contentsOf: url)
            tokens = try JSONDecoder().decode([Int32].self, from: data)
            inputSource = ["kind": "token_file", "path": url.path,
                           "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
        } else {
            let text = "请用三句话介绍太阳为什么发光，以及阳光如何到达地球。"
            tokens = try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: text)]))
            inputSource = ["kind": "tokenized_user_text", "text": text, "chat_template": true]
        }
        guard tokens.count <= 128 else { throw CLIError.usage("MTP state probe requires at most 128 prompt tokens") }
        let configuration = try QwenConfiguration(modelDirectory: directory)
        try QwenGenerationRequest(tokens: tokens, maxTokens: 9, contextLimit: 4096)
            .validate(configuration: configuration)
        var previousCacheLimit = 0
        try MX.check(mlx_set_cache_limit(&previousCacheLimit, 256 * 1024 * 1024), "bound MTP state probe cache")
        defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try QwenModel(modelDirectory: directory) { current, total in
            if current % 8 == 0 || current == total {
                FileHandle.standardError.write(Data("MTP state probe: loaded \(current)/\(total) layers\n".utf8))
            }
        }
        let trunkLoadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) * 1e-9
        let headStart = DispatchTime.now().uptimeNanoseconds
        let head = try QwenMTP(weights: model.weights, configuration: configuration)
        let headLoadSeconds = Double(DispatchTime.now().uptimeNanoseconds - headStart) * 1e-9
        var checks = [[String: Any]](), details = [String: Any]()
        func record(_ id: String, _ status: String, _ values: [String: Any] = [:]) {
            var row = values
            row["id"] = id; row["status"] = status
            row["required"] = true; row["passed"] = status == "passed"
            checks.append(row)
            FileHandle.standardError.write(Data("MTP state probe: \(id): \(status)\n".utf8))
        }
        func json<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func writeReport() throws {
            try emit([
                "schema": "qwen38-mtp-state-stop-probe-v1",
                "model_directory": directory.path, "input": inputSource,
                "prompt_token_ids": tokens, "full_model_instances": 1, "mtp_head_instances": 1,
                "budget_eos_verification": verification.rawValue,
                "cancel_after_verify_policy": cancellationVerification.rawValue,
                "cancel_after_history_policy": historyCancellationVerification.rawValue,
                "trunk_construction_seconds": trunkLoadSeconds, "head_construction_seconds": headLoadSeconds,
                "passed": checks.allSatisfy { $0["status"] as? String == "passed" },
                "checks": checks, "details": details,
                "notes": [
                    "Cancellation checks the caller's offset, validity, ordered Tensor wrapper identities and MLX handle identities; tensor bytes are not copied back or hashed.",
                    "Cancellation is injected at callback 4 in cancel_after_verify_policy depth 1, after target verification/prefix commit and before publication.",
                    "Budget/EOS verification is named in budget_eos_verification; this does not qualify other modes or contexts.",
                    "The nine-token raw AR oracle disables EOS stopping to provide enough deterministic continuation IDs for these bounded tests.",
                    "EOS tests designate a known target continuation ID as the stop token. These are real decoder stop branches, not evidence of a naturally generated model EOS.",
                    "At most seven real AR positions are searched for accepted/rejected draft EOS coverage. A missing branch is not_exercised, never passed.",
                    "No draft proposals or model outputs are injected or altered. This is a correctness gate, not a throughput benchmark."
                ]
            ], to: output)
        }

        func prepare(_ prompt: [Int32]) throws -> GPUMTPPreparedPrompt {
            var state = model.makeState(), offset = 0, pending: Int32 = 0
            var parts = [(offset: Int, stream: Tensor)]()
            let prefetch = try model.makePrefillPrefetch(tokens: prompt, chunk: 416, state: state)
            defer { prefetch.finish() }
            while offset < prompt.count {
                let end = offset < prompt.count - 1 ? min(prompt.count - 1, offset + 416) : prompt.count
                let out = try model.forward(tokens: Array(prompt[offset..<end]), state: &state, prefillPrefetch: prefetch)
                if end == prompt.count {
                    guard let logits = out.logits else { throw CLIError.usage("Missing target logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected, out.stream], state: &state)
                    pending = try selected.uint32TokenID()
                } else { try model.evaluate([out.stream], state: &state) }
                parts.append((offset, out.stream)); offset = end
            }
            return GPUMTPPreparedPrompt(tokens: prompt, state: state, pending: pending, parts: parts)
        }
        func decoder(_ prepared: GPUMTPPreparedPrompt,
                     verification override: QwenMTPDecoder.Verification? = nil) throws -> QwenMTPDecoder {
            let decoder = QwenMTPDecoder(model: model, head: head, verification: override ?? verification)
            for part in prepared.parts {
                try decoder.consumePrompt(stream: part.stream, prompt: prepared.tokens, offset: part.offset)
            }
            return decoder
        }
        func rawAR(_ prepared: GPUMTPPreparedPrompt, count: Int) throws -> [Int32] {
            var state = prepared.state, output = [prepared.pending]
            while output.count < count {
                let out = try model.forward(tokens: [output.last!], state: &state)
                guard let logits = out.logits else { throw CLIError.usage("Missing AR logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected], state: &state)
                output.append(try selected.uint32TokenID())
            }
            return output
        }
        func refused(_ decoder: QwenMTPDecoder, state: inout QwenModel.State,
                     pending: Int32) -> (passed: Bool, error: String) {
            var callbacks = 0
            do {
                _ = try decoder.next(pending: pending, state: &state, depth: 1, remaining: 2, eos: [],
                                     checkCancellation: { callbacks += 1 })
                return (false, "Decoder unexpectedly returned a second result")
            } catch {
                if case GPUError.invalid(let message) = error {
                    return (callbacks == 0 && message == "Invalid MTP generation position or budget", message)
                }
                return (false, String(describing: error))
            }
        }

        do {
            let base = try prepare(tokens)
            let reference = try rawAR(base, count: 9)
            details["raw_ar_reference_token_ids"] = reference
            record("raw_ar_reference", "passed", ["generated_token_count": reference.count])

            // The callback count is checked against actual verified-token stats,
            // so a future check insertion cannot silently move this pre-verify.
            do {
                let candidate = try decoder(base, verification: cancellationVerification)
                var state = base.state, calls = 0
                let before = GPUMTPStateIdentity(state)
                var capturedError: Error?
                do {
                    _ = try candidate.next(pending: base.pending, state: &state, depth: 1, remaining: 3, eos: [],
                                           checkCancellation: {
                        calls += 1
                        if calls == 4 { throw QwenGenerationError.cancelled }
                    })
                } catch { capturedError = error }
                try MX.synchronize()
                let after = GPUMTPStateIdentity(state)
                let verified = candidate.statistics.verifiedTokens > 0
                let passed = capturedError as? QwenGenerationError == .cancelled && calls == 4 && verified && before == after
                record("cancel_after_verify_preserves_caller_state", passed ? "passed" : "failed", [
                    "check_calls": calls, "error": capturedError.map { String(describing: $0) } ?? NSNull(),
                    "verification_exercised": verified, "identity_exact": before == after,
                    "before": try json(before), "after": try json(after),
                    "statistics": try json(candidate.statistics)
                ])
                let reuse = refused(candidate, state: &state, pending: base.pending)
                record("cancelled_decoder_refuses_reuse", reuse.passed && GPUMTPStateIdentity(state) == before ? "passed" : "failed",
                       ["error": reuse.error, "caller_state_still_exact": GPUMTPStateIdentity(state) == before])
            } catch { record("cancel_after_verify_preserves_caller_state", "failed", ["error": String(describing: error)]) }

            // Observe cancellation again at the final publication boundary,
            // after the last target/head operation has evaluated successfully.
            for (name, remaining, cancelAt, mode) in [
                ("cancel_after_final_target", 1, 2, QwenMTPDecoder.Verification.scalar),
                ("cancel_after_head_history", 3, 5, historyCancellationVerification)
            ] {
                do {
                    let candidate = try decoder(base, verification: mode)
                    var state = base.state, calls = 0
                    let before = GPUMTPStateIdentity(state)
                    var captured: Error?
                    do {
                        _ = try candidate.next(pending: base.pending, state: &state, depth: 1,
                            remaining: remaining, eos: [], checkCancellation: {
                                calls += 1
                                if calls == cancelAt { throw QwenGenerationError.cancelled }
                            })
                    } catch { captured = error }
                    try MX.synchronize()
                    let reachedPublication = candidate.statistics.verifiedTokens > 0 &&
                        (remaining == 1 || candidate.statistics.rounds == 1)
                    let reuse = refused(candidate, state: &state, pending: base.pending)
                    record(name, captured as? QwenGenerationError == .cancelled && calls == cancelAt &&
                        reachedPublication && GPUMTPStateIdentity(state) == before && reuse.passed ? "passed" : "failed",
                        ["check_calls": calls, "publication_boundary_reached": reachedPublication,
                         "caller_identity_exact": GPUMTPStateIdentity(state) == before,
                         "decoder_refuses_reuse": reuse.passed, "statistics": try json(candidate.statistics)])
                } catch { record(name, "failed", ["error": String(describing: error)]) }
            }

            do {
                let retried = try rawAR(prepare(tokens), count: reference.count)
                record("fresh_ar_after_cancel", retried == reference ? "passed" : "failed", ["token_ids": retried])
            } catch { record("fresh_ar_after_cancel", "failed", ["error": String(describing: error)]) }

            for depth in [1, 2] {
                for budget in [1, 2, 3] {
                    let name = "depth_\(depth)_remaining_\(budget)"
                    do {
                        let candidate = try decoder(base)
                        var state = base.state, pending = base.pending, emitted = [Int32](), rounds = [[String: Any]]()
                        while emitted.count < budget {
                            let remaining = budget - emitted.count
                            let round = try candidate.next(pending: pending, state: &state, depth: depth, remaining: remaining, eos: [])
                            guard !round.tokens.isEmpty, round.tokens.count <= remaining else {
                                throw CLIError.usage("Decoder exceeded or failed to advance its remaining budget")
                            }
                            emitted.append(contentsOf: round.tokens); pending = round.tokens.last!
                            rounds.append(["remaining_before": remaining, "emitted_ids": round.tokens,
                                           "state_offset": state.offset])
                        }
                        let expected = Array(reference.dropFirst().prefix(budget))
                        let finalIdentity = GPUMTPStateIdentity(state)
                        let reuse = refused(candidate, state: &state, pending: pending)
                        let passed = emitted == expected && emitted.count == budget &&
                            state.offset == base.state.offset + budget && reuse.passed &&
                            GPUMTPStateIdentity(state) == finalIdentity
                        record(name, passed ? "passed" : "failed", [
                            "expected_ids": expected, "emitted_ids": emitted, "rounds": rounds,
                            "final_state_offset": state.offset, "exhausted_decoder_refuses_reuse": reuse.passed,
                            "reuse_error": reuse.error, "statistics": try json(candidate.statistics)
                        ])
                    } catch { record(name, "failed", ["error": String(describing: error)]) }
                }
            }

            // Observe the real draft branch before selecting the test EOS ID.
            // A fresh decoder replays the same prompt; no proposals are mocked.
            var context = base, covered = Set<String>(), searchRows = [[String: Any]]()
            for position in 0..<7 {
                if covered.count == 2 { break }
                do {
                    let observation = try decoder(context)
                    var observedState = context.state
                    let observed = try observation.next(pending: context.pending, state: &observedState,
                                                        depth: 1, remaining: 2, eos: [])
                    let expected = Array(reference.dropFirst(position + 1).prefix(observed.tokens.count))
                    guard observed.tokens == expected, let stop = observed.tokens.first else {
                        throw CLIError.usage("Scalar EOS branch discovery differed from the raw AR oracle")
                    }
                    let branch = observation.statistics.acceptedDraftTokens > 0 ? "accepted_draft_eos" : "correction_eos"
                    searchRows.append(["position": position, "pending_token": context.pending,
                                       "observed_ids": observed.tokens, "branch": branch,
                                       "statistics": try json(observation.statistics)])
                    if !covered.contains(branch) {
                        let stopping = try decoder(context)
                        var state = context.state
                        let stopped = try stopping.next(pending: context.pending, state: &state,
                                                        depth: 1, remaining: 3, eos: [stop])
                        let branchMatches = branch == "accepted_draft_eos"
                            ? stopping.statistics.acceptedDraftTokens == 1 : stopping.statistics.acceptedDraftTokens == 0
                        let terminal = GPUMTPStateIdentity(state)
                        let reuse = refused(stopping, state: &state, pending: stop)
                        let passed = stopped.tokens == [stop] && branchMatches && reuse.passed &&
                            GPUMTPStateIdentity(state) == terminal
                        record(branch, passed ? "passed" : "failed", [
                            "ar_position": position, "test_stop_token_id": stop, "emitted_ids": stopped.tokens,
                            "natural_eos_generation_test": false, "stop_policy": "synthetic_known_target_token",
                            "token_id_is_native_eos": tokenizer.eosTokenIDs.contains(stop), "remaining_before": 3,
                            "terminal_decoder_refuses_reuse": reuse.passed, "reuse_error": reuse.error,
                            "statistics": try json(stopping.statistics)
                        ])
                        covered.insert(branch)
                    }
                    if covered.count == 2 || position == 6 { break }
                    let offset = context.state.offset, consumed = context.pending
                    let out = try model.forward(tokens: [consumed], state: &context.state)
                    guard let logits = out.logits else { throw CLIError.usage("Missing AR context-advance logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected, out.stream], state: &context.state)
                    context.pending = try selected.uint32TokenID()
                    guard context.pending == reference[position + 1] else { throw CLIError.usage("AR context advance changed tokens") }
                    context.tokens.append(consumed); context.parts.append((offset, out.stream))
                } catch {
                    record("eos_branch_search", "failed", ["position": position, "error": String(describing: error)])
                    break
                }
            }
            details["eos_branch_search"] = searchRows
            for branch in ["accepted_draft_eos", "correction_eos"] where !covered.contains(branch) {
                record(branch, "not_exercised", ["reason": "The bounded real-proposal search did not observe this branch"])
            }
        } catch {
            record("probe_setup_or_reference", "failed", ["error": String(describing: error)])
            try writeReport()
            throw error
        }
        try MX.synchronize()
        try writeReport()
        guard checks.allSatisfy({ $0["status"] as? String == "passed" }) else {
            throw CLIError.usage("MTP state probe has failed or unexercised checks; see \(output)")
        }
    }
}

private struct GPUMTPPreparedPrompt {
    var tokens: [Int32]
    var state: QwenModel.State
    var pending: Int32
    var parts: [(offset: Int, stream: Tensor)]
}

private struct GPUMTPStateIdentity: Codable, Equatable {
    let offset: Int
    let valid: Bool
    let qsaActiveLayers: Int
    let tensorWrappers: [String]
    let mlxHandles: [String]
    let shapes: [[Int]]
    let dtypes: [String]
    init(_ state: QwenModel.State) {
        let tensors = state.tensors
        offset = state.offset; valid = state.valid; qsaActiveLayers = state.qsaActiveLayers
        tensorWrappers = tensors.map { String(describing: ObjectIdentifier($0)) }
        mlxHandles = tensors.map { String(describing: $0.handle.ctx) }
        shapes = tensors.map(\.shape)
        dtypes = tensors.map { String(describing: $0.dtype) }
    }
}
