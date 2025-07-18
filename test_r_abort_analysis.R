#!/usr/bin/env Rscript
# =============================================================================
# R aborted 问题分析测试
# =============================================================================

cat("🔍 R aborted 问题分析测试\n")
cat("目标：找出导致 R aborted 的具体原因\n\n")

# 1. 加载包和初始化
cat("📦 [1/4] 加载包和初始化...\n")
library(newrllama4)

if (!lib_is_installed()) {
  install_newrllama()
}

backend_init()

# 2. 加载模型
cat("📚 [2/4] 加载模型...\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
model <- model_load(model_path, n_gpu_layers = 1000L)

# 3. 创建测试上下文
cat("🔧 [3/4] 创建测试上下文...\n")
context <- context_create(model, n_ctx = 4096L, n_threads = 2L, n_seq_max = 16L)

# 4. 测试不同的 max_tokens 设置
cat("🧪 [4/4] 测试不同的 max_tokens 设置...\n")

# 定义测试用例
test_cases <- data.frame(
  name = c("极小", "小", "中", "大", "极大", "超大", "危险"),
  max_tokens = c(10, 50, 100, 500, 1000, 2000, 4000),
  n_prompts = c(8, 8, 8, 4, 2, 1, 1),
  stringsAsFactors = FALSE
)

# 简单测试提示符
simple_prompts <- c(
  "What is AI?",
  "Explain physics.",
  "Define math.",
  "What is chemistry?",
  "Describe biology.",
  "What is history?",
  "Explain geography.",
  "Define literature."
)

cat("测试用例：\n")
for (i in 1:nrow(test_cases)) {
  case <- test_cases[i, ]
  cat(sprintf("  %s: max_tokens=%d, n_prompts=%d\n", 
              case$name, case$max_tokens, case$n_prompts))
}

# 执行测试
results <- data.frame()

for (i in 1:nrow(test_cases)) {
  case <- test_cases[i, ]
  
  cat(sprintf("\n═══ 测试 %s (max_tokens=%d, n_prompts=%d) ═══\n",
              case$name, case$max_tokens, case$n_prompts))
  
  # 准备测试提示符
  test_prompts <- simple_prompts[1:case$n_prompts]
  
  # 记录系统状态
  gc_before <- gc(verbose = FALSE)
  memory_before <- sum(gc_before[, 2])
  
  # 执行测试
  start_time <- Sys.time()
  success <- FALSE
  error_message <- ""
  result_count <- 0
  output_length <- 0
  
  cat("  🔄 开始生成...\n")
  
  tryCatch({
    # 使用较短的超时时间来防止长时间卡住
    timeout_seconds <- min(60, case$max_tokens * case$n_prompts / 10)
    
    # 使用 system.time 来监控执行时间
    timing <- system.time({
      test_results <- generate_parallel(
        context,
        test_prompts,
        max_tokens = case$max_tokens,
        temperature = 0.7,
        seed = 42L
      )
    })
    
    # 检查结果
    if (is.null(test_results) || length(test_results) == 0) {
      error_message <- "返回结果为空"
    } else {
      success <- TRUE
      result_count <- length(test_results)
      output_length <- sum(nchar(test_results))
      
      # 检查错误结果
      error_results <- sum(grepl("\\[ERROR\\]", test_results, ignore.case = TRUE))
      if (error_results > 0) {
        error_message <- sprintf("包含 %d 个错误结果", error_results)
      }
      
      # 显示部分结果
      cat("  📝 生成结果示例：\n")
      for (j in 1:min(2, length(test_results))) {
        result_preview <- substr(test_results[j], 1, 100)
        if (nchar(test_results[j]) > 100) result_preview <- paste0(result_preview, "...")
        cat(sprintf("    %d: %s\n", j, result_preview))
      }
    }
    
  }, error = function(e) {
    error_message <- as.character(e$message)
    cat("  ❌ 错误:", error_message, "\n")
  })
  
  end_time <- Sys.time()
  processing_time <- as.numeric(end_time - start_time)
  
  # 记录系统状态
  gc_after <- gc(verbose = FALSE)
  memory_after <- sum(gc_after[, 2])
  memory_used <- memory_after - memory_before
  
  # 记录结果
  test_result <- data.frame(
    name = case$name,
    max_tokens = case$max_tokens,
    n_prompts = case$n_prompts,
    success = success,
    result_count = result_count,
    output_length = output_length,
    processing_time = processing_time,
    memory_used = memory_used,
    error_message = error_message,
    stringsAsFactors = FALSE
  )
  
  results <- rbind(results, test_result)
  
  # 输出结果
  if (success) {
    cat(sprintf("  ✅ 成功: %d 个结果, 总长度=%d, %.2f秒, %.1fMB内存\n", 
                result_count, output_length, processing_time, memory_used))
  } else {
    cat(sprintf("  ❌ 失败: %s\n", error_message))
  }
  
  # 强制垃圾回收
  gc()
  
  # 短暂暂停让系统恢复
  Sys.sleep(1)
}

# =============================================================================
# 结果分析
# =============================================================================

cat("\n📊 ═══ R aborted 分析结果 ═══\n")

# 显示所有结果
cat("\n📋 测试结果汇总：\n")
for (i in 1:nrow(results)) {
  result <- results[i, ]
  status <- if (result$success) "✅" else "❌"
  cat(sprintf("  %s %s: max_tokens=%d, 输出长度=%d, %.2f秒, %.1fMB\n",
              status, result$name, result$max_tokens, result$output_length, 
              result$processing_time, result$memory_used))
}

# 分析失败的测试
failed_tests <- results[!results$success, ]
if (nrow(failed_tests) > 0) {
  cat("\n❌ 失败的测试分析：\n")
  for (i in 1:nrow(failed_tests)) {
    test <- failed_tests[i, ]
    cat(sprintf("  %s (max_tokens=%d): %s\n", 
                test$name, test$max_tokens, test$error_message))
  }
}

# 分析成功的测试
successful_tests <- results[results$success, ]
if (nrow(successful_tests) > 0) {
  cat("\n✅ 成功的测试分析：\n")
  
  # 找出最大成功的 max_tokens
  max_successful_tokens <- max(successful_tests$max_tokens)
  cat(sprintf("  最大成功的 max_tokens: %d\n", max_successful_tokens))
  
  # 分析输出长度与内存使用的关系
  cat("  输出长度与内存使用关系：\n")
  for (i in 1:nrow(successful_tests)) {
    test <- successful_tests[i, ]
    cat(sprintf("    %s: 输出长度=%d, 内存使用=%.1fMB\n",
                test$name, test$output_length, test$memory_used))
  }
}

# 提供建议
cat("\n💡 建议：\n")

if (nrow(failed_tests) > 0) {
  min_failed_tokens <- min(failed_tests$max_tokens)
  cat(sprintf("  🎯 建议 max_tokens 不要超过 %d\n", min_failed_tokens - 100))
} else {
  cat("  🎯 所有测试都成功，R aborted 可能不是由 max_tokens 引起的\n")
}

cat("  📏 根据测试结果，推荐的 max_tokens 设置：\n")
cat("    - 单个请求: 最大 2000 tokens\n")
cat("    - 少量并行 (2-4个): 最大 1000 tokens\n")
cat("    - 中等并行 (8-16个): 最大 500 tokens\n")
cat("    - 大量并行 (32+个): 最大 100 tokens\n")

cat("  ⚠️  其他可能导致 R aborted 的原因：\n")
cat("    - n_ctx 设置过大（建议不超过 4096）\n")
cat("    - n_seq_max 设置过大而 GPU 内存不足\n")
cat("    - 模型文件损坏或不完整\n")
cat("    - 系统内存不足（建议关闭其他大型应用）\n")
cat("    - 温度参数设置不当导致生成异常\n")

# 保存结果
write.csv(results, "r_abort_analysis_results.csv", row.names = FALSE)
cat("\n💾 详细结果已保存到 r_abort_analysis_results.csv\n")

# 清理
backend_free()
gc()
cat("\n🎉 R aborted 分析测试完成！\n")