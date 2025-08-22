# 专门调试apply_chat_template问题
library(newrllama4)

cat("=== 调试apply_chat_template返回空结果问题 ===\n\n")

# 初始化（使用最小配置加快速度）
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
cat("加载模型:", model_path, "\n")

model <- model_load(model_path, verbosity = 0)
cat("模型加载完成\n")

# 测试1: 检查apply_chat_template的不同调用方式
cat("\n--- 测试1: apply_chat_template详细调试 ---\n")

messages <- list(
  list(role = "user", content = "Hello")
)

# 方式1: 使用NULL（自动使用模型template）
cat("方式1: 使用NULL template\n")
result1 <- apply_chat_template(model, messages, template = NULL)
cat("结果1长度:", nchar(result1), "\n")
cat("结果1内容:", dQuote(result1), "\n")

# 方式2: 使用空字符串
cat("\n方式2: 使用空字符串template\n")
result2 <- apply_chat_template(model, messages, template = "")
cat("结果2长度:", nchar(result2), "\n")
cat("结果2内容:", dQuote(result2), "\n")

# 方式3: 使用基本Gemma格式template  
cat("\n方式3: 使用手动Gemma template\n")
gemma_template <- "<start_of_turn>user\n{{{ content }}}<end_of_turn>\n<start_of_turn>model\n"
result3 <- apply_chat_template(model, messages, template = gemma_template)
cat("结果3长度:", nchar(result3), "\n")
cat("结果3内容:", dQuote(result3), "\n")

# 测试2: 检查messages格式是否正确
cat("\n--- 测试2: 检查messages格式 ---\n")
cat("Messages结构:\n")
str(messages)

# 测试3: 简化消息测试
cat("\n--- 测试3: 最简单的消息测试 ---\n")
simple_messages <- list(list(role = "user", content = "Hi"))
result_simple <- apply_chat_template(model, simple_messages, template = NULL)
cat("简单消息结果长度:", nchar(result_simple), "\n")
cat("简单消息结果:", dQuote(result_simple), "\n")

# 测试4: 检查错误处理
cat("\n--- 测试4: 错误处理测试 ---\n")
tryCatch({
  bad_result <- apply_chat_template(model, list(), template = NULL)
  cat("空消息列表结果:", dQuote(bad_result), "\n")
}, error = function(e) {
  cat("空消息列表出错:", e$message, "\n")
})

# 清理
backend_free()
cat("\n=== apply_chat_template调试完成 ===\n")