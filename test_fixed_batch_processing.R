library(newrllama4)

cat("=== 测试修复后的批处理功能 ===\n")
cat("配置: 5个prompt, 3个seq_max (小于prompt数量)\n\n")

# 加载模型
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf",
  n_gpu_layers = 20,
  verbosity = 1
)

# 创建上下文 - 故意设置seq_max小于prompt数量
ctx <- context_create(model, n_ctx = 2000, n_seq_max = 3)

# 准备5个简单的prompt
prompts <- c(
  "Answer: What is 2+2?",
  "Answer: What is the capital of France?",
  "Answer: What color is the sky?",
  "Answer: What is 5*3?",
  "Answer: Name a fruit."
)

cat("准备了", length(prompts), "个prompt，seq_max =", 3, "\n")
cat("开始并行生成...\n")

start_time <- Sys.time()

tryCatch({
  results <- generate_parallel(
    context = ctx,
    prompts = prompts,
    max_tokens = 5L,
    temperature = 0.1,
    top_k = 10L,
    top_p = 0.9,
    repeat_last_n = 16L,
    penalty_repeat = 1.1,
    seed = 1234L
  )

  end_time <- Sys.time()
  duration <- as.numeric(end_time - start_time, units = "secs")

  cat("✅ 成功完成! 耗时:", round(duration, 2), "秒\n")
  cat("结果:\n")
  for (i in 1:length(results)) {
    cat(sprintf("%d. %s -> %s\n", i, substr(prompts[i], 1, 30), trimws(results[i])))
  }

}, error = function(e) {
  cat("❌ 错误:", e$message, "\n")
})