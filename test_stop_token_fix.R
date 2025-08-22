# 测试停止标记泄漏修复效果
library(newrllama4)

cat("=== 测试停止标记泄漏修复效果 ===\n\n")

# 初始化并加载模型
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, verbosity = 0)
ctx <- context_create(model, n_ctx = 512, verbosity = 0)

# 测试1: 单个生成函数（应该已经正确）
cat("--- 测试1: 单个生成函数 ---\n")
messages1 <- list(
  list(role = "user", content = "What is the square root of 144? Reply with only the number.")
)
formatted_prompt1 <- apply_chat_template(model, messages1)
tokens1 <- tokenize(model, formatted_prompt1)
result1 <- generate(ctx, tokens1, max_tokens = 10)

cat("单个生成结果: '", result1, "'\n", sep = "")
cat("包含<end_of_turn>: ", grepl("<end_of_turn>", result1), "\n")
cat("包含停止标记: ", grepl("<|im_end|>|<end_of_turn>|</s>|<eos>", result1), "\n\n")

# 测试2: 并行生成函数（应该被修复）
cat("--- 测试2: 并行生成函数（修复后）---\n")
test_prompts <- c(
  "What is 2+2? Answer with just the number.",
  "Name one color. Just the color name.",
  "What is the capital of Japan? Just the city name."
)

# 格式化所有prompts
formatted_prompts <- sapply(test_prompts, function(prompt) {
  messages <- list(list(role = "user", content = prompt))
  apply_chat_template(model, messages)
})

# 并行生成
results2 <- generate_parallel(ctx, formatted_prompts, max_tokens = 8, temperature = 0.1, seed = 42)

cat("并行生成结果:\n")
for(i in 1:length(results2)) {
  clean_result <- trimws(results2[i])
  cat("结果", i, ": '", clean_result, "'\n", sep = "")
  cat("  包含<end_of_turn>: ", grepl("<end_of_turn>", clean_result), "\n")
  cat("  包含任何停止标记: ", grepl("<|im_end|>|<end_of_turn>|</s>|<eos>", clean_result), "\n")
}

# 测试3: 原始问题重现（144平方根）
cat("\n--- 测试3: 原始问题重现 ---\n")
messages3 <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "What is the square root of 144? Reply with only the number.")
)
formatted_prompt3 <- apply_chat_template(model, messages3)
tokens3 <- tokenize(model, formatted_prompt3)
result3 <- generate(ctx, tokens3, max_tokens = 10)

cat("原始问题结果: '", result3, "'\n", sep = "")
cat("期望: '12'\n")
cat("实际得到: '", result3, "'\n", sep = "")
cat("修复成功: ", !grepl("<end_of_turn>", result3) && grepl("12", result3), "\n")

# 清理
backend_free()
cat("\n=== 停止标记修复测试完成 ===\n")