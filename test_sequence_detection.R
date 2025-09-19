#!/usr/bin/env Rscript

# 专门测试multi-token序列检测逻辑
library(newrllama4)

cat("=== 测试v1.0.77的Multi-token序列检测 ===\n\n")

if (!lib_is_installed()) {
  install_newrllama()
}

model_path <- "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

cat("📥 加载模型和上下文...\n")
model <- model_load(model_path, n_gpu_layers = 100L, verbosity = 0L)
ctx <- context_create(model, n_ctx = 512L, verbosity = 0L)

# 测试关键：专门构造一个会产生我们已知序列的场景
cat("\n🎯 关键测试：验证序列检测逻辑\n")
cat(strrep("=", 50), "\n")

# 我们已知的EOG序列：
# <|eot_id|> = [27, 91, 68, 354, 851, 91, 29]
cat("已知EOG序列：\n")
cat("  <|eot_id|> = [27, 91, 68, 354, 851, 91, 29]\n\n")

# 使用一个简单的prompt来测试
simple_prompt <- "Hello"
tokens_in <- tokenize(model, simple_prompt)
cat(sprintf("测试prompt: '%s'\n", simple_prompt))
cat(sprintf("输入tokens: [%s]\n", paste(tokens_in, collapse = ", ")))

# 生成tokens并观察是否被我们的序列检测捕获
cat("\n📤 生成测试 (max_tokens=100):\n")
result <- generate(model, ctx, tokens_in, max_tokens = 100L)

cat(sprintf("生成结果: '%s'\n", result))
cat(sprintf("结果长度: %d 字符\n", nchar(result)))

# 检查是否包含完整的EOG序列
eog_found <- FALSE
if (grepl("<\\|eot_id\\|>", result)) {
  eog_found <- TRUE
  cat("\n❌ 发现完整的 <|eot_id|> 在输出中\n")
  cat("   这意味着我们的序列检测逻辑没有生效\n")
}

if (grepl("<\\|end_header_id\\|>", result)) {
  eog_found <- TRUE
  cat("\n❌ 发现完整的 <|end_header_id|> 在输出中\n")
  cat("   这意味着我们的序列检测逻辑没有生效\n")
}

if (!eog_found) {
  cat("\n✅ 没有发现完整的EOG序列\n")
  cat("   序列检测可能正在工作，或者这个prompt不会触发EOG\n")
}

# 额外测试：尝试用更可能触发EOG的prompt
cat("\n🧪 额外测试：使用聊天格式prompt\n")
cat(strrep("-", 40), "\n")

# 使用会触发聊天模板的prompt
chat_result <- quick_llama("Tell me a joke", max_tokens = 50)
cat(sprintf("quick_llama结果: '%s'\n", chat_result))

if (grepl("<\\|eot_id\\|>", chat_result)) {
  cat("❌ quick_llama最终输出仍包含<|eot_id|>\n")
  cat("   混合策略完全失败\n")
} else {
  cat("✅ quick_llama输出干净\n")
  cat("   R层清理生效（但可能C++层仍有问题）\n")
}

cat("\n📊 诊断结论：\n")
if (eog_found) {
  cat("1. ❌ Multi-token序列检测逻辑没有被执行\n")
  cat("2. 🔧 可能的原因：\n")
  cat("   - 编译问题：新代码没有被包含在预编译库中\n")
  cat("   - 逻辑错误：token序列匹配条件不正确\n")
  cat("   - 执行路径问题：代码路径没有被调用\n")
} else {
  cat("1. ✅ 当前测试场景下序列检测可能工作\n")
  cat("2. 🔧 需要更多测试来确认\n")
}

rm(model, ctx)
backend_free()
cat("\n✅ 测试完成\n")