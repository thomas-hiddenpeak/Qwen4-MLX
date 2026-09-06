import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// One loaded model, short real requests; this is a lifecycle and token
    /// equivalence gate, not a throughput benchmark or broad quality test.
    static func probeGPUSession(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output"])
        let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Session probe --output must be a new file")
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
            let prompt = "请用三句话介绍太阳为什么发光，以及阳光如何到达地球。"
            tokens = try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: prompt)]))
            inputSource = ["kind": "tokenized_user_text", "text": prompt,
                           "chat_template": true, "thinking": false]
        }
        let configuration = try QwenConfiguration(modelDirectory: directory)
        let maximumOutput = 8
        let (budget, overflow) = tokens.count.addingReportingOverflow(maximumOutput)
        guard !overflow, budget <= configuration.maximumPositions else {
            throw CLIError.usage("Session probe prompt plus eight output tokens exceeds model context")
        }
        let context = max(4096, budget)
        let request = QwenGenerationRequest(tokens: tokens, maxTokens: maximumOutput, contextLimit: context)
        // Reject an unusable probe input before loading the full model. The
        // explicit invalid-request cases below exercise the generator itself.
        try request.validate(configuration: configuration)
        var previousCacheLimit = 0
        try MX.check(mlx_set_cache_limit(&previousCacheLimit, 256 * 1024 * 1024), "bound session probe allocation cache")
        defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
        let constructionStart = DispatchTime.now().uptimeNanoseconds
        let model = try QwenModel(modelDirectory: directory) { current, total in
            if current % 8 == 0 || current == total {
                FileHandle.standardError.write(Data("Session probe: loaded layer \(current)/\(total)\n".utf8))
            }
        }
        let constructionSeconds = Double(DispatchTime.now().uptimeNanoseconds - constructionStart) * 1e-9
        let generator = try QwenGenerator(model: model)
        let nestedGenerator = try QwenGenerator(model: model)
        var checks = [[String: Any]](), results = [String: Any]()
        func record(_ id: String, _ status: String, _ details: [String: Any] = [:]) {
            var row = details
            row["id"] = id; row["status"] = status; row["required"] = true
            row["passed"] = status == "passed"
            checks.append(row)
            FileHandle.standardError.write(Data("Session probe: \(id): \(status)\n".utf8))
        }
        func save(_ name: String, _ result: QwenGenerationResult) throws {
            results[name] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result))
        }
        func writeReport() throws {
            try emit([
                "schema": "qwen38-typed-session-probe-v1", "model_directory": directory.path,
                "passed": checks.allSatisfy { $0["status"] as? String == "passed" },
                "input": inputSource, "prompt_token_ids": tokens,
                "max_output_tokens": maximumOutput, "context_limit": context,
                "full_model_instances": 1, "generator_instances": 2,
                "model_construction_seconds": constructionSeconds,
                "loaded_source_weight_bytes_at_report": model.weights.cachedSourceBytes,
                "checks": checks, "results": results,
                "optional_checks": [["id": "mtp_batched", "status": "not_run",
                                      "reason": "This bounded probe only requires scalar depth 1 and 2; no batched equivalence is inferred."]],
                "notes": [
                    "All requests reuse one full model and create fresh request state; cancellation/error retries use the same generator.",
                    "Equality covers emitted token IDs, finish reason and reported final offset, not all internal tensors.",
                    "MTP uses the native Qwen4 head with scalar target verification. Acceptance/rejection branch coverage is not inferred from requested depth.",
                    "Per-request timings include different first-use and cache conditions; this is not a performance comparison.",
                    "A callback that is never reached is recorded as not_exercised, never passed."
                ]
            ], to: output)
        }
        func validResult(_ result: QwenGenerationResult) -> Bool {
            let n = result.tokens.count
            return n > 0 && n <= maximumOutput &&
                result.statistics.generatedTokenCount == n &&
                result.statistics.promptTokenCount == tokens.count &&
                result.statistics.finalStateOffset == tokens.count + n - 1 &&
                result.statistics.decodedTokenCount == n - 1 &&
                (result.finishReason == .length ? n == maximumOutput : tokenizer.eosTokenIDs.contains(result.tokens.last!))
        }

        let reference: QwenGenerationResult
        do {
            reference = try generator.generate(request)
            try save("ar_reference", reference)
            record("ar_reference", validResult(reference) ? "passed" : "failed", ["state_offset_consistent": validResult(reference)])
        } catch {
            record("ar_reference", "failed", ["error": String(describing: error)])
            try writeReport()
            throw error
        }
        func exact(_ result: QwenGenerationResult) -> Bool {
            validResult(result) && result.tokens == reference.tokens &&
                result.finishReason == reference.finishReason &&
                result.statistics.finalStateOffset == reference.statistics.finalStateOffset
        }
        func retry(_ name: String) {
            do {
                let result = try generator.generate(request)
                try save(name, result)
                record(name, exact(result) ? "passed" : "failed", ["tokens_exact": result.tokens == reference.tokens])
            } catch { record(name, "failed", ["error": String(describing: error)]) }
        }

        let cancellation = QwenCancellation()
        var cancelledIDs = [Int32](), cancellationError: Error?
        do {
            let result = try generator.generate(request, cancellation: cancellation) { token in
                cancelledIDs.append(token)
                if cancelledIDs.count == 2 { cancellation.cancel() }
            }
            try save("cancellation_unexpected_result", result)
        } catch { cancellationError = error }
        let cancellationReached = cancelledIDs.count >= 2
        let cancelledCorrectly = cancellationError as? QwenGenerationError == .cancelled &&
            cancelledIDs.count == 2 && cancelledIDs == Array(reference.tokens.prefix(2))
        record("cancel_at_second_token", cancellationReached ? (cancelledCorrectly ? "passed" : "failed") : "not_exercised",
               ["callback_tokens": cancelledIDs, "error": cancellationError.map { String(describing: $0) } ?? NSNull()])
        retry("fresh_request_after_cancel")

        var errorIDs = [Int32](), callbackError: Error?
        do {
            let result = try generator.generate(request) { token in
                errorIDs.append(token)
                if errorIDs.count == 2 { throw GPUSessionProbeCallbackError.deliberate }
            }
            try save("callback_error_unexpected_result", result)
        } catch { callbackError = error }
        let callbackReached = errorIDs.count >= 2
        let callbackCorrect = callbackError as? GPUSessionProbeCallbackError == .deliberate &&
            errorIDs.count == 2 && errorIDs == Array(reference.tokens.prefix(2))
        record("callback_error_propagates", callbackReached ? (callbackCorrect ? "passed" : "failed") : "not_exercised",
               ["callback_tokens": errorIDs, "error": callbackError.map { String(describing: $0) } ?? NSNull()])
        retry("fresh_request_after_callback_error")

        var nestedAttempted = false, nestedError: Error?, nestedReturned = false
        var nestedSeconds: Double?
        do {
            let result = try generator.generate(request) { _ in
                guard !nestedAttempted else { return }
                nestedAttempted = true
                let start = DispatchTime.now().uptimeNanoseconds
                do { _ = try nestedGenerator.generate(request); nestedReturned = true }
                catch { nestedError = error }
                nestedSeconds = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
            }
            try save("outer_request_after_nested_attempt", result)
            record("nested_request_same_model_busy", nestedAttempted && !nestedReturned && nestedError as? QwenGenerationError == .busy ? "passed" : "failed",
                   ["attempted": nestedAttempted, "returned_result": nestedReturned,
                    "error": nestedError.map { String(describing: $0) } ?? NSNull(),
                    "call_wall_seconds": nestedSeconds.map { $0 as Any } ?? NSNull()])
            record("outer_request_survives_busy", exact(result) ? "passed" : "failed")
        } catch {
            record("nested_request_same_model_busy", "failed", ["attempted": nestedAttempted, "error": String(describing: error)])
            record("outer_request_survives_busy", "failed", ["error": String(describing: error)])
        }

        let invalid: [(String, QwenGenerationRequest)] = [
            ("empty_prompt_rejected", .init(tokens: [], maxTokens: maximumOutput, contextLimit: context)),
            ("negative_token_rejected", .init(tokens: [-1], maxTokens: maximumOutput, contextLimit: context)),
            ("out_of_vocabulary_rejected", .init(tokens: [Int32(configuration.vocabularySize)], maxTokens: maximumOutput, contextLimit: context)),
            ("context_budget_rejected", .init(tokens: tokens, maxTokens: maximumOutput, contextLimit: budget - 1))
        ]
        for (name, badRequest) in invalid {
            var callbacks = 0
            do {
                _ = try generator.generate(badRequest) { _ in callbacks += 1 }
                record(name, "failed", ["reason": "Invalid request unexpectedly generated", "callbacks": callbacks])
            } catch {
                let rejected: Bool
                if case .invalidRequest = error as? QwenGenerationError { rejected = true }
                else { rejected = false }
                record(name, rejected && callbacks == 0 ? "passed" : "failed",
                       ["error": String(describing: error), "callbacks": callbacks])
            }
        }
        retry("fresh_request_after_invalid_requests")

        for depth in [1, 2] {
            let name = "mtp_scalar_depth_\(depth)"
            let mtpRequest = QwenGenerationRequest(tokens: tokens, maxTokens: maximumOutput,
                contextLimit: context, mtpDepth: depth, verification: .scalar)
            do {
                let result = try generator.generate(mtpRequest)
                try save(name, result)
                let drafted = result.statistics.mtp?.draftedTokens ?? 0
                let exercised = drafted > 0
                let matches = exact(result) && result.statistics.mtpDepth == depth &&
                    result.statistics.mtpVerification == "scalar"
                record(name, exercised ? (matches ? "passed" : "failed") : "not_exercised",
                       ["tokens_exact": result.tokens == reference.tokens, "drafted_tokens": drafted,
                        "accepted_draft_tokens": result.statistics.mtp?.acceptedDraftTokens ?? 0])
            } catch { record(name, "failed", ["error": String(describing: error)]) }
        }
        // Also prove that lazy MTP preparation did not change subsequent AR.
        retry("ar_after_mtp_requests")
        try writeReport()
        guard checks.allSatisfy({ $0["status"] as? String == "passed" }) else {
            throw CLIError.usage("Session probe has failed or unexercised required checks; see \(output)")
        }
    }
}

private enum GPUSessionProbeCallbackError: Error, Equatable { case deliberate }
