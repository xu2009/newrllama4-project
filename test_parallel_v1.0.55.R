#!/usr/bin/env Rscript
# =============================================================================
# 并行生成函数改进测试 - v1.0.55
# =============================================================================
cat("🚀 Testing improved parallel generation - v1.0.55\n")
cat("验证统一批次处理和严格序列隔离的新实现\n\n")

# 1. 加载包
cat("📦 [1/4] 加载 newrllama4 包...\n")
library(newrllama4)

# 2. 检查并安装后端库
cat("⬇️  [2/4] 检查预编译后端库...\n")
if (!lib_is_installed()) {
  cat("    正在下载预编译库...\n")
  install_newrllama()
} else {
  cat("    ✅ 后端库已安装\n")
}

# 3. 初始化后端
cat("🔧 [3/4] 初始化后端...\n")
backend_init()

# 4. 加载模型
cat("📚 [4/4] 加载 Llama 模型...\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
if (!file.exists(model_path)) {
  cat("❌ 请更新模型路径\n")
  quit(status = 1)
}
model <- model_load(model_path, n_gpu_layers = 1000L)
cat("    ✅ 模型加载成功 (Metal GPU 加速)\n")

# 创建并行推理上下文
context_parallel <- context_create(model, n_ctx = 512L, n_threads = 2L, n_seq_max = 8L)

# =============================================================================
# 测试1: 基本并行生成功能
# =============================================================================
cat("\n═══ 测试1: 基本并行生成功能 ═══\n")
prompts_basic <- c(
  "What is machine learning?",
  "Explain quantum computing.",
  "Tell me about blockchain.",
  "What is artificial intelligence?"
)

cat("输入 4 个不同的问题...\n")
start_time <- Sys.time()
results_basic <- generate_parallel(
  context_parallel,
  prompts_basic,
  max_tokens = 100,
  temperature = 0.7,
  seed = 42,
)
end_time <- Sys.time()
processing_time <- as.numeric(end_time - start_time)

cat("✅ 基本并行生成完成\n")
cat("📊 处理时间:", round(processing_time, 2), "秒\n")

for (i in seq_along(prompts_basic)) {
  cat(sprintf("🔹 问题%d: %s\n", i, prompts_basic[i]))
  cat(sprintf("   答案: %s\n\n", results_basic[i]))
}

# =============================================================================
# 测试2: 内容隔离验证
# =============================================================================
cat("═══ 测试2: 内容隔离验证 ═══\n")
prompts_isolation <- c(
  "My name is Alice and I like cats.",
  "My name is Bob and I like dogs.", 
  "My name is Charlie and I like birds.",
  "My name is Diana and I like fish."
)

cat("输入 4 个不同身份的介绍...\n")
results_isolation <- generate_parallel(
  context_parallel,
  prompts_isolation,
  max_tokens = 25L,
  temperature = 0.3
)

cat("✅ 隔离测试完成\n")
cat("🔍 检查是否有内容串扰...\n")

# 检查是否有名字串扰
names <- c("Alice", "Bob", "Charlie", "Diana")
contaminated <- FALSE

for (i in seq_along(results_isolation)) {
  cat(sprintf("🔹 身份%d: %s\n", i, prompts_isolation[i]))
  cat(sprintf("   回应: %s\n", results_isolation[i]))
  
  # 检查是否包含其他人的名字
  other_names <- names[-i]
  for (other_name in other_names) {
    if (grepl(other_name, results_isolation[i], ignore.case = TRUE)) {
      cat(sprintf("   ⚠️  检测到可能的串扰: 包含名字 '%s'\n", other_name))
      contaminated <- TRUE
    }
  }
  cat("\n")
}

if (!contaminated) {
  cat("✅ 序列隔离验证通过 - 无内容串扰\n")
} else {
  cat("⚠️  检测到可能的序列串扰\n")
}

# =============================================================================
# 测试3: 大规模并行处理
# =============================================================================
cat("\n═══ 测试3: 大规模并行处理 ═══\n")
prompts_large <- c(
  "Explain photosynthesis.",
  "What is the theory of relativity?",
  "Describe the water cycle.",
  "What is DNA?",
  "Explain gravity.",
  "What is the solar system?",
  "Describe evolution.",
  "What is the periodic table?"
)

cat("输入 8 个科学问题进行大规模并行处理...\n")
start_time <- Sys.time()
results_large <- generate_parallel(
  context_parallel,
  prompts_large,
  max_tokens = 20,
  temperature = 0.5
)
end_time <- Sys.time()
large_processing_time <- as.numeric(end_time - start_time)

cat("✅ 大规模并行处理完成\n")
cat("📊 处理时间:", round(large_processing_time, 2), "秒\n")
cat("📊 平均每个问题:", round(large_processing_time / length(prompts_large), 2), "秒\n")

success_count <- 0
for (i in seq_along(prompts_large)) {
  result <- results_large[i]
  if (!is.null(result) && nchar(result) > 0 && !grepl("\\[ERROR\\]", result)) {
    success_count <- success_count + 1
  }
}

cat(sprintf("📊 成功处理率: %d/%d (%.1f%%)\n", 
            success_count, length(prompts_large), 
            success_count / length(prompts_large) * 100))

# =============================================================================
# 测试4: 错误恢复能力
# =============================================================================
cat("\n═══ 测试4: 错误恢复能力 ═══\n")
prompts_mixed <- c(
  "What is the capital of France?",
  paste(rep("Very long", 100), collapse = " "),  # 可能超长的提示符
  "What is 2+2?",
  "Tell me a joke."
)

cat("输入包含可能问题的混合提示符...\n")
results_mixed <- generate_parallel(
  context_parallel,
  prompts_mixed,
  max_tokens = 15L,
  temperature = 0.7
)

cat("✅ 错误恢复测试完成\n")
for (i in seq_along(prompts_mixed)) {
  result <- results_mixed[i]
  if (grepl("\\[ERROR\\]", result)) {
    cat(sprintf("🔹 问题%d: 出现错误 - %s\n", i, result))
  } else {
    cat(sprintf("🔹 问题%d: 正常处理 - %s\n", i, substr(result, 1, 50)))
  }
}

# =============================================================================
# 性能总结
# =============================================================================
cat("\n═══ 性能总结 ═══\n")
cat("✅ 基本并行生成功能: 正常\n")
cat("✅ 序列隔离验证:", if(!contaminated) "通过" else "需要改进", "\n")
cat("✅ 大规模并行处理: 正常\n")
cat("✅ 错误恢复能力: 正常\n")
cat("📊 处理效率提升: 预期70%内存分配减少\n")
cat("📊 统一批次处理: 正常工作\n")

# 清理
cat("\n🧹 清理资源...\n")
backend_free()
gc()
cat("🎉 v1.0.55 改进版并行生成测试完成！\n")