import Foundation

/// Narrow OpenAI-compatible JSON/SSE bodies. IDs, created and actual usage are
/// supplied by the request owner, so every event shares one request identity.
public enum QwenHTTPFrames {
    public static func role(id: String, created: Int, model: String) throws -> Data {
        try event(chunk(id: id, created: created, model: model, delta: ["role": "assistant"], reason: NSNull()))
    }
    public static func content(id: String, created: Int, model: String, text: String) throws -> Data {
        try event(chunk(id: id, created: created, model: model, delta: ["content": text], reason: NSNull()))
    }
    /// One final completion chunk. Append done() to build a terminal payload.
    public static func finish(id: String, created: Int, model: String, reason: String,
                              promptTokens: Int, completionTokens: Int) throws -> Data {
        try validateReason(reason)
        var result = chunk(id: id, created: created, model: model, delta: [:], reason: reason)
        result["usage"] = try usage(promptTokens: promptTokens, completionTokens: completionTokens)
        return try event(result)
    }
    public static func done() -> Data { Data("data: [DONE]\n\n".utf8) }

    public static func completion(id: String, created: Int, model: String, text: String, reason: String,
                                  promptTokens: Int, completionTokens: Int) throws -> Data {
        try validateReason(reason)
        let result: [String: Any] = ["id": id, "object": "chat.completion", "created": created, "model": model,
            "choices": [["index": 0, "message": ["role": "assistant", "content": text], "finish_reason": reason]],
            "usage": try usage(promptTokens: promptTokens, completionTokens: completionTokens)]
        return try json(result)
    }
    public static func error(message: String, code: String, type: String = "invalid_request_error") throws -> Data {
        try json(errorObject(message: message, code: code, type: type))
    }
    public static func sseError(message: String, code: String, type: String = "invalid_request_error") throws -> Data {
        try event(errorObject(message: message, code: code, type: type))
    }
    public static func models(model: String, created: Int) throws -> Data {
        try json(["object": "list", "data": [["id": model, "object": "model", "created": created, "owned_by": "local"]]])
    }

    public static func response(status: Int, contentType: String, body: Data) throws -> Data {
        guard (100...599).contains(status), !contentType.isEmpty,
              contentType.utf8.allSatisfy({ (32...126).contains($0) }) else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Invalid response header")
        }
        let descriptions = [200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 408: "Request Timeout",
            411: "Length Required", 413: "Content Too Large", 415: "Unsupported Media Type",
            417: "Expectation Failed", 429: "Too Many Requests", 431: "Request Header Fields Too Large",
            500: "Internal Server Error", 503: "Service Unavailable"]
        let reason = descriptions[status] ?? "Response"
        var result = Data("HTTP/1.1 \(status) \(reason)\r\nContent-Type: \(contentType)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
        result.append(body)
        return result
    }
    /// A close-delimited SSE body: no Content-Length and no Transfer-Encoding.
    public static func streamHeader() -> Data {
        Data("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n".utf8)
    }

    private static func chunk(id: String, created: Int, model: String, delta: [String: String], reason: Any) -> [String: Any] {
        ["id": id, "object": "chat.completion.chunk", "created": created, "model": model,
         "choices": [["index": 0, "delta": delta, "finish_reason": reason]]]
    }
    private static func errorObject(message: String, code: String, type: String) -> [String: Any] {
        ["error": ["message": message, "type": type, "param": NSNull(), "code": code]]
    }
    private static func usage(promptTokens: Int, completionTokens: Int) throws -> [String: Int] {
        let (total, overflow) = promptTokens.addingReportingOverflow(completionTokens)
        guard promptTokens >= 0, completionTokens >= 0, !overflow else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Invalid actual token usage")
        }
        return ["prompt_tokens": promptTokens, "completion_tokens": completionTokens, "total_tokens": total]
    }
    private static func validateReason(_ reason: String) throws {
        guard ["stop", "length"].contains(reason) else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Unsupported finish reason")
        }
    }
    private static func json(_ object: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
    }
    private static func event(_ object: [String: Any]) throws -> Data {
        var result = Data("data: ".utf8)
        result.append(try json(object)); result.append(Data("\n\n".utf8))
        return result
    }
}
