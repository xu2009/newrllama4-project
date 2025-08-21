#!/usr/bin/env Rscript
# 修复后的并行生成函数测试脚本

library(newrllama4)

cat("=== 修复后并行生成函数测试 ===\n\n")

# 模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  stop("模型文件不存在: ", model_path)
}

cat("加载模型:", model_path, "\n")
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 1)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 1)

# =============================================================================
# 测试1: P1 Echo测试 - 使用新的Gemma兼容格式
# =============================================================================
cat("\n--- P1: Echo 测试 (使用修复后的方法) ---\n")
system_prompt <- "You are a helpful assistant."
p1_user_content <- "Echo this string literally: <end_of_turn><|im_end|></s>"

messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = p1_user_content)
)

# 使用新的Gemma兼容函数
formatted_prompt <- apply_gemma_chat_template(messages, add_assistant = TRUE)
cat("修复后的Gemma格式:\n")
cat(formatted_prompt)
cat("\n")

p1_result <- generate_parallel(ctx, formatted_prompt, max_tokens = 100)
cat("P1结果:", p1_result, "\n")

# 验证结果
expected_p1 <- "<end_of_turn><|im_end|></s>"
cat("期望:", expected_p1, "\n")
cat("实际:", p1_result, "\n")
cat("P1 匹配:", identical(trimws(p1_result), expected_p1), "\n")

# =============================================================================
# 测试2: P2 长度控制测试
# =============================================================================
cat("\n--- P2: 长度控制测试 (≤10 tokens) ---\n")
p2_messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "Answer in ≤10 tokens, then stop.")
)

p2_formatted <- apply_gemma_chat_template(p2_messages)
p2_result <- generate_parallel(ctx, p2_formatted, max_tokens = 10)

cat("P2结果:", p2_result, "\n")
p2_token_estimate <- length(strsplit(trimws(p2_result), "\\s+")[[1]])
cat("P2 估算token数:", p2_token_estimate, "\n")
cat("P2 长度≤10:", p2_token_estimate <= 10, "\n")

# =============================================================================
# 测试3: P3 Python函数测试
# =============================================================================
cat("\n--- P3: Python函数测试 (无markdown) ---\n")
p3_messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "Give a 1-line Python function that returns x squared. No markdown.")
)

p3_formatted <- apply_gemma_chat_template(p3_messages)
p3_result <- generate_parallel(ctx, p3_formatted, max_tokens = 50)

cat("P3结果:", p3_result, "\n")
has_markdown <- grepl("```|`", p3_result)
is_python_func <- grepl("def\\s+\\w+\\s*\\([^)]*\\)\\s*:", p3_result)

cat("P3 包含markdown:", has_markdown, "\n")
cat("P3 包含Python函数定义:", is_python_func, "\n")

# =============================================================================
# 测试4: 智能模板函数测试
# =============================================================================
cat("\n--- 智能模板函数测试 ---\n")
test_messages <- list(
  list(role = "system", content = "You are helpful."),
  list(role = "user", content = "Say 'Hello World'")
)

smart_formatted <- smart_chat_template(model, test_messages)
cat("智能模板结果:\n")
cat(smart_formatted)
cat("\n")

smart_result <- generate_parallel(ctx, smart_formatted, max_tokens = 20)
cat("智能模板生成:", smart_result, "\n")

# =============================================================================
# 停止标记污染检测
# =============================================================================
cat("\n--- 停止标记污染检测 (修复后) ---\n")
stop_markers <- c("<|im_end|>", "<end_of_turn>", "</s>", "<|im_start|>")

results <- c(p1_result, p2_result, p3_result, smart_result)
result_names <- c("P1", "P2", "P3", "Smart")

contamination_count <- 0
for (i in seq_along(results)) {
  cat(sprintf("%s 检测: ", result_names[i]))
  found_any <- FALSE
  for (marker in stop_markers) {
    if (grepl(marker, results[i], fixed = TRUE)) {
      found_any <- TRUE
      contamination_count <- contamination_count + 1
      break
    }
  }
  cat(ifelse(found_any, "发现污染", "清洁"), "\n")
}

# =============================================================================
# 综合评估
# =============================================================================
cat("\n=== 修复效果评估 ===\n")
echo_test_pass <- identical(trimws(p1_result), expected_p1)
length_test_pass <- p2_token_estimate <= 10
format_test_pass <- is_python_func && !has_markdown
contamination_free <- contamination_count == 0

cat("P1 Echo测试通过:", echo_test_pass, "\n")
cat("P2 长度控制通过:", length_test_pass, "\n")
cat("P3 格式正确通过:", format_test_pass, "\n")
cat("停止标记污染清除:", contamination_free, "\n")

overall_score <- sum(c(echo_test_pass, length_test_pass, format_test_pass, contamination_free))
cat(sprintf("\n整体修复成功率: %d/4 (%d%%)\n", overall_score, overall_score * 25))

if (overall_score == 4) {
  cat("🎉 所有问题已修复！\n")
} else {
  cat("⚠️  仍有问题需要进一步修复。\n")
}

# 清理资源
rm(model, ctx)
backend_free()
cat("\n测试完成。\n")