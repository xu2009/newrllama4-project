#!/usr/bin/env Rscript
# 并行生成函数质量测试脚本
# 用于验证用户报告的P1、P2、P3问题

library(newrllama4)

cat("=== 并行生成函数质量测试 ===\n\n")

# 模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

# 检查模型文件
if (!file.exists(model_path)) {
  stop("模型文件不存在: ", model_path)
}

cat("加载模型:", model_path, "\n")
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 2)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 2)

# =============================================================================
# 测试1: P1 Echo测试 - 期望逐字回显，无额外字符
# =============================================================================
cat("\n--- P1: Echo 测试 ---\n")
system_prompt <- "You are a helpful assistant."

# 测试用例P1：逐字回显特殊字符串
p1_user_content <- "Echo this string literally: <end_of_turn><|im_end|></s>"

# 方法1：使用apply_chat_template
cat("方法1 - 使用apply_chat_template:\n")
p1_messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = p1_user_content)
)
p1_formatted <- apply_chat_template(model, p1_messages)
cat("生成的chat template:\n")
cat(p1_formatted)
cat("\n")

p1_result_template <- generate_parallel(ctx, p1_formatted, max_tokens = 100)
cat("P1结果(template):", p1_result_template, "\n")

# 方法2：手动构造Gemma格式
cat("\n方法2 - 手动构造Gemma格式:\n")
p1_manual_format <- paste0(
  "<start_of_turn>user\n",
  system_prompt, "\n\n", p1_user_content,
  "<end_of_turn>\n<start_of_turn>model\n"
)
cat("手动构造的格式:\n")
cat(p1_manual_format)
cat("\n")

p1_result_manual <- generate_parallel(ctx, p1_manual_format, max_tokens = 100)
cat("P1结果(manual):", p1_result_manual, "\n")

# 分析P1结果
expected_p1 <- "<end_of_turn><|im_end|></s>"
cat("\n期望:", expected_p1, "\n")
cat("实际(template):", p1_result_template, "\n")
cat("实际(manual):", p1_result_manual, "\n")
cat("P1 Template匹配:", identical(trimws(p1_result_template), expected_p1), "\n")
cat("P1 Manual匹配:", identical(trimws(p1_result_manual), expected_p1), "\n")

# =============================================================================
# 测试2: P2 长度控制测试 - ≤10 tokens
# =============================================================================
cat("\n--- P2: 长度控制测试 (≤10 tokens) ---\n")
p2_user_content <- "Answer in ≤10 tokens, then stop."

# 使用较小的max_tokens设置
p2_messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = p2_user_content)
)
p2_formatted <- apply_chat_template(model, p2_messages)
p2_result <- generate_parallel(ctx, p2_formatted, max_tokens = 10)  # 严格限制10 tokens

cat("P2结果:", p2_result, "\n")

# 简单的token计数估算（空格分割）
p2_token_estimate <- length(strsplit(trimws(p2_result), "\\s+")[[1]])
cat("P2 估算token数:", p2_token_estimate, "\n")
cat("P2 长度是否≤10:", p2_token_estimate <= 10, "\n")

# =============================================================================
# 测试3: P3 Python函数测试 - 无markdown
# =============================================================================
cat("\n--- P3: Python函数测试 (无markdown) ---\n")
p3_user_content <- "Give a 1-line Python function that returns x squared. No markdown."

p3_messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = p3_user_content)
)
p3_formatted <- apply_chat_template(model, p3_messages)
p3_result <- generate_parallel(ctx, p3_formatted, max_tokens = 50)

cat("P3结果:", p3_result, "\n")

# 检查是否包含markdown符号
has_markdown <- grepl("```|`", p3_result)
cat("P3 包含markdown:", has_markdown, "\n")

# 检查是否是有效的Python函数格式
is_python_func <- grepl("def\\s+\\w+\\s*\\([^)]*\\)\\s*:", p3_result)
cat("P3 包含Python函数定义:", is_python_func, "\n")

# =============================================================================
# 停止标记污染检测
# =============================================================================
cat("\n--- 停止标记污染检测 ---\n")
stop_markers <- c("<|im_end|>", "<end_of_turn>", "</s>", "<|im_start|>", "\\n", "\\r")

results <- c(p1_result_template, p1_result_manual, p2_result, p3_result)
result_names <- c("P1_template", "P1_manual", "P2", "P3")

for (i in seq_along(results)) {
  cat(sprintf("%s 停止标记检测:\n", result_names[i]))
  for (marker in stop_markers) {
    has_marker <- grepl(marker, results[i], fixed = TRUE)
    if (has_marker) {
      cat(sprintf("  - 发现 '%s': %s\n", marker, has_marker))
    }
  }
  cat("\n")
}

# =============================================================================
# 综合评估
# =============================================================================
cat("=== 综合评估 ===\n")
cat("P1 Echo测试通过:", identical(trimws(p1_result_manual), expected_p1), "\n")
cat("P2 长度控制通过:", p2_token_estimate <= 10, "\n")
cat("P3 格式正确通过:", is_python_func && !has_markdown, "\n")

# 清理资源
rm(model, ctx)
backend_free()
cat("\n测试完成，资源已清理。\n")