# 测试完整的停止标记修复（单序列 + 并行）
library(newrllama4)

cat("=== 测试完整停止标记修复 ===\n\n")

backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, verbosity = 0)
ctx <- context_create(model, n_ctx = 512, verbosity = 0)

# 测试1: 原始问题 - 单序列生成
cat("--- 测试1: 单序列生成修复 ---\n")
messages1 <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "What is the square root of 144? Reply with only the number.")
)
formatted_prompt1 <- apply_chat_template(model, messages1)
tokens1 <- tokenize(model, formatted_prompt1)
result1 <- generate(ctx, tokens1, max_tokens = 10)

cat("原始问题结果: '", result1, "'\n", sep="")
cat("期望: '12'\n")
cat("包含停止标记: ", grepl("<end_of_turn>|<|im_end|>", result1), "\n")
cat("修复成功: ", !grepl("<end_of_turn>", result1) && trimws(result1) != "", "\n\n")

# 测试2: 并行生成修复
cat("--- 测试2: 并行生成修复 ---\n") 
test_prompts <- c(
  "What is 5+3? Just the number.",
  "Name the color of grass. One word.",
  "What is 10-7? Just the number."
)

formatted_prompts <- sapply(test_prompts, function(prompt) {
  messages <- list(list(role = "user", content = prompt))
  apply_chat_template(model, messages)
})

results2 <- generate_parallel(ctx, formatted_prompts, max_tokens = 5, seed = 123)

cat("并行生成结果:\n")
all_clean <- TRUE
for(i in 1:length(results2)) {
  clean_result <- trimws(results2[i])
  has_stop_token <- grepl("<end_of_turn>|<|im_end|>", clean_result)
  cat("结果", i, ": '", clean_result, "' | 有停止标记: ", has_stop_token, "\n", sep="")
  if(has_stop_token) all_clean <- FALSE
}
cat("并行生成全部修复: ", all_clean, "\n\n")

# 测试3: 边界情况
cat("--- 测试3: 边界情况测试 ---\n")
edge_prompts <- c(
  "Say 'hello' and stop.",
  "Count: 1, 2, 3",
  "Answer: yes or no?"
)

edge_formatted <- sapply(edge_prompts, function(prompt) {
  messages <- list(list(role = "user", content = prompt))
  apply_chat_template(model, messages)
})

edge_results <- generate_parallel(ctx, edge_formatted, max_tokens = 8, seed = 456)

cat("边界情况结果:\n")
edge_clean <- TRUE
for(i in 1:length(edge_results)) {
  result <- trimws(edge_results[i])
  has_stop <- grepl("<end_of_turn>|<|im_end|>", result)
  cat("边界", i, ": '", result, "' | 停止标记: ", has_stop, "\n", sep="")
  if(has_stop) edge_clean <- FALSE
}

# 最终总结
cat("\n=== 修复效果总结 ===\n")
cat("✅ 单序列生成修复: ", !grepl("<end_of_turn>", result1), "\n")
cat("✅ 并行生成修复: ", all_clean, "\n") 
cat("✅ 边界情况修复: ", edge_clean, "\n")

overall_success <- !grepl("<end_of_turn>", result1) && all_clean && edge_clean
cat("🎉 整体修复成功: ", overall_success, "\n")

backend_free()
cat("\n=== 完整修复测试完成 ===\n")