# 快速验证v1.0.69修复效果
library(newrllama4)

cat("=== 快速验证v1.0.69修复效果 ===\n\n")

# 初始化
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
cat("加载模型:", model_path, "\n")
model <- model_load(model_path, verbosity = 0)
ctx <- context_create(model, n_ctx = 512, verbosity = 0)

# 测试1: apply_chat_template是否能正常使用模型内置template
cat("\n--- 测试1: apply_chat_template自动使用模型template ---\n")
messages <- list(
  list(role = "user", content = "简单回答：1+1等于多少？")
)

# 使用NULL template（应该自动使用模型内置template）
result1 <- apply_chat_template(model, NULL, messages)
cat("使用NULL template结果长度:", nchar(result1), "\n")
cat("前100字符:", substr(result1, 1, 100), "\n")

# 测试2: 并行生成基本功能
cat("\n--- 测试2: 并行生成基本功能 ---\n")
prompts <- c(
  "写一个数字: ",
  "说一个颜色: ",
  "说一个动物: "
)

start_time <- Sys.time()
results <- generate_parallel(
  ctx = ctx,
  prompts = prompts,
  max_new_tokens = 10,
  temperature = 0.7,
  seed = 42
)
end_time <- Sys.time()

cat("并行生成耗时:", as.numeric(end_time - start_time, units = "secs"), "秒\n")
for(i in 1:length(results)) {
  cat("结果", i, ":", trimws(results[i]), "\n")
}

# 测试3: 验证独立性（种子控制）
cat("\n--- 测试3: 验证种子控制独立性 ---\n")
result_a1 <- generate_parallel(ctx, c("数字: "), max_new_tokens = 5, seed = 123)
result_a2 <- generate_parallel(ctx, c("数字: "), max_new_tokens = 5, seed = 123)
result_b <- generate_parallel(ctx, c("数字: "), max_new_tokens = 5, seed = 456)

cat("种子123结果1:", trimws(result_a1[1]), "\n")
cat("种子123结果2:", trimws(result_a2[1]), "\n") 
cat("种子456结果: ", trimws(result_b[1]), "\n")
cat("相同种子一致性:", identical(result_a1[1], result_a2[1]), "\n")
cat("不同种子差异性:", !identical(result_a1[1], result_b[1]), "\n")

# 清理
backend_free()
cat("\n=== 快速验证完成 ===\n")