import Foundation

/// The shipped Qwen3.8 chat template's thinking-disabled text/function arm.
/// Tool results are a single user turn containing consecutive response blocks.
public enum QwenToolChatTemplate {
    public struct Message: Sendable {
        public let role: String, content: String
        public let reasoningContent: String?
        public let calls: [QwenToolCall]
        public init(role: String, content: String, reasoningContent: String? = nil, calls: [QwenToolCall] = []) {
            self.role = role; self.content = content; self.reasoningContent = reasoningContent; self.calls = calls
        }
    }
    private static let instructions = """


If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""
    public static func systemPrefix(system: String?, tools: [QwenToolDefinition]) -> String {
        let content = trim(system ?? "")
        if tools.isEmpty { return content.isEmpty ? "" : "<|im_start|>system\n" + content + "<|im_end|>\n" }
        var text = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for tool in tools { text += "\n" + tool.wire.templateJSON() }
        text += "\n</tools>" + instructions
        if !content.isEmpty { text += "\n\n" + content }
        return text + "<|im_end|>\n"
    }
    public static func render(messages: [Message], tools: [QwenToolDefinition], addGenerationPrompt: Bool = true) throws -> String {
        guard messages.contains(where: { $0.role == "user" && !(trim($0.content).hasPrefix("<tool_response>") && trim($0.content).hasSuffix("</tool_response>")) }) else {
            throw QwenToolDefinition.invalid("Chat requires a user query")
        }
        var text = systemPrefix(system: messages.first?.role == "system" ? messages.first?.content : nil, tools: tools)
        for (index, message) in messages.enumerated() {
            let content = trim(message.content)
            switch message.role {
            case "system":
                guard index == 0 else { throw QwenToolDefinition.invalid("System message must be first") }
            case "user": text += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "assistant":
                text += "<|im_start|>assistant\n<think>\n" + trim(message.reasoningContent ?? "") + "\n</think>\n\n" + content
                for (callIndex, call) in message.calls.enumerated() {
                    if callIndex > 0 { text += "\n" }
                    else if !content.isEmpty { text += "\n\n" }
                    text += "<tool_call>\n<function=" + call.name + ">\n"
                    for key in (call.arguments.object ?? [:]).keys.sorted() {
                        let value = call.arguments.object![key]!
                        text += "<parameter=" + key + ">\n" + (value.string ?? value.templateJSON()) + "\n</parameter>\n"
                    }
                    text += "</function>\n</tool_call>"
                }
                text += "<|im_end|>\n"
            case "tool":
                guard index > 0 else { throw QwenToolDefinition.invalid("Tool results need an assistant call") }
                if messages[index - 1].role != "tool" { text += "<|im_start|>user" }
                text += "\n<tool_response>\n" + content + "\n</tool_response>"
                if index == messages.count - 1 || messages[index + 1].role != "tool" { text += "<|im_end|>\n" }
            default: throw QwenToolDefinition.invalid("Unsupported chat role")
            }
        }
        if addGenerationPrompt { text += "<|im_start|>assistant\n<think>\n\n</think>\n\n" }
        return text
    }
    private static func trim(_ value: String) -> String { value.trimmingCharacters(in: .whitespacesAndNewlines) }
}
