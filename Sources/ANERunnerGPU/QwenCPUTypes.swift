import ANERunnerCore

// Preserve source compatibility for existing MLX clients after moving the
// CPU-only tokenizer and configuration into the backend-independent module.
public typealias QwenConfiguration = ANERunnerCore.QwenConfiguration
public typealias QwenTokenizer = ANERunnerCore.QwenTokenizer
public typealias ChatMessage = ANERunnerCore.ChatMessage
public typealias QwenConversationPrefixPlan = ANERunnerCore.QwenConversationPrefixPlan
public typealias QwenTokenizedConversation = ANERunnerCore.QwenTokenizedConversation
