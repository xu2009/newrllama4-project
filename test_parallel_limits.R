#!/usr/bin/env Rscript
# =============================================================================
# 大规模并行生成上限测试 - 找出瓶颈所在
# =============================================================================

cat("🔍 大规模并行生成上限测试 - v1.0.55\n")
cat("目标：找出模型、电脑、R Studio、并行函数的具体限制\n\n")

# 1. 加载包和初始化
cat("📦 [1/4] 加载包和初始化...\n")
library(newrllama4)

if (!lib_is_installed()) {
  cat("    正在下载预编译库...\n")
  install_newrllama()
}

backend_init()

# 2. 加载模型
cat("📚 [2/4] 加载模型...\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
if (!file.exists(model_path)) {
  cat("❌ 模型文件不存在\n")
  quit(status = 1)
}

model <- model_load(model_path, n_gpu_layers = 1000L)
cat("    ✅ 模型加载成功\n")

# 3. 创建不同配置的上下文进行测试
cat("🔧 [3/4] 创建测试上下文...\n")

# 测试不同的n_seq_max配置
test_configs <- data.frame(
  name = c("小型", "中型", "大型", "超大型"),
  n_seq_max = c(8, 16, 32, 64),
  n_ctx = c(512, 1024, 2048, 4096),
  stringsAsFactors = FALSE
)

cat("    配置选项：\n")
for (i in 1:nrow(test_configs)) {
  cat(sprintf("    %s: n_seq_max=%d, n_ctx=%d\n", 
              test_configs$name[i], test_configs$n_seq_max[i], test_configs$n_ctx[i]))
}

# 4. 生成测试提示符
cat("📝 [4/4] 准备测试提示符...\n")

# 创建大量简短的测试提示符
base_prompts <- c(
  "What is AI?",
  "Explain physics.",
  "Define mathematics.",
  "What is chemistry?",
  "Describe biology.",
  "What is history?",
  "Explain geography.",
  "Define literature.",
  "What is music?",
  "Describe art.",
  "What is philosophy?",
  "Explain psychology.",
  "Define sociology.",
  "What is economics?",
  "Describe politics.",
  "What is law?",
  "Explain medicine.",
  "Define engineering.",
  "What is technology?",
  "Describe science."
)

# 生成不同规模的测试集
test_scales <- c(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)

cat("    生成的测试规模：", paste(test_scales, collapse = ", "), "\n")

# =============================================================================
# 主要测试循环
# =============================================================================

cat("\n🚀 开始大规模并行测试...\n")

# 测试结果存储
results <- data.frame()

# 对每个配置进行测试
for (config_idx in 1:nrow(test_configs)) {
  config <- test_configs[config_idx, ]
  
  cat(sprintf("\n═══ 测试配置: %s (n_seq_max=%d, n_ctx=%d) ═══\n",
              config$name, config$n_seq_max, config$n_ctx))
  
  # 创建上下文
  tryCatch({
    context <- context_create(model, 
                             n_ctx = config$n_ctx, 
                             n_threads = 2L, 
                             n_seq_max = config$n_seq_max)
    cat("    ✅ 上下文创建成功\n")
  }, error = function(e) {
    cat("    ❌ 上下文创建失败:", e$message, "\n")
    next
  })
  
  # 对每个规模进行测试
  for (scale in test_scales) {
    if (scale > config$n_seq_max) {
      cat(sprintf("    ⏭️  跳过 %d 个请求 (超出 n_seq_max=%d)\n", scale, config$n_seq_max))
      next
    }
    
    cat(sprintf("    🔄 测试 %d 个并行请求...\n", scale))
    
    # 生成测试提示符
    test_prompts <- rep(base_prompts, length.out = scale)
    if (scale > length(base_prompts)) {
      # 为更大规模添加编号
      test_prompts <- paste0(test_prompts, " (", 1:scale, ")")
    }
    
    # 记录系统状态
    gc_before <- gc()
    memory_before <- sum(gc_before[, 2])
    
    # 执行测试
    start_time <- Sys.time()
    success <- FALSE
    error_message <- ""
    result_count <- 0
    
    tryCatch({
      test_results <- generate_parallel(
        context,
        test_prompts,
        max_tokens = 10L,  # 使用短输出减少内存压力
        temperature = 0.5,
        seed = 42L
      )
      
      # 检查结果
      if (is.null(test_results) || length(test_results) == 0) {
        error_message <- "返回结果为空"
      } else {
        success <- TRUE
        result_count <- length(test_results)
        
        # 检查错误结果
        error_results <- sum(grepl("\\[ERROR\\]", test_results, ignore.case = TRUE))
        if (error_results > 0) {
          error_message <- sprintf("包含 %d 个错误结果", error_results)
        }
      }
      
    }, error = function(e) {
      error_message <- as.character(e$message)
    })
    
    end_time <- Sys.time()
    processing_time <- as.numeric(end_time - start_time)
    
    # 记录系统状态
    gc_after <- gc()
    memory_after <- sum(gc_after[, 2])
    memory_used <- memory_after - memory_before
    
    # 记录结果
    test_result <- data.frame(
      config_name = config$name,
      n_seq_max = config$n_seq_max,
      n_ctx = config$n_ctx,
      request_count = scale,
      success = success,
      result_count = result_count,
      processing_time = processing_time,
      memory_used = memory_used,
      error_message = error_message,
      stringsAsFactors = FALSE
    )
    
    results <- rbind(results, test_result)
    
    # 输出结果
    if (success) {
      cat(sprintf("    ✅ 成功: %d/%d 结果, %.2f秒, %.1fMB内存\n", 
                  result_count, scale, processing_time, memory_used))
    } else {
      cat(sprintf("    ❌ 失败: %s\n", error_message))
    }
    
    # 如果失败，停止测试更大规模
    if (!success) {
      cat(sprintf("    🛑 配置 %s 在 %d 个请求时达到上限\n", config$name, scale))
      break
    }
    
    # 清理内存
    gc()
    Sys.sleep(0.5)  # 短暂暂停让系统恢复
  }
}

# =============================================================================
# 结果分析
# =============================================================================

cat("\n📊 ═══ 测试结果分析 ═══\n")

# 显示所有成功的测试
successful_tests <- results[results$success, ]
if (nrow(successful_tests) > 0) {
  cat("\n✅ 成功的测试：\n")
  for (i in 1:nrow(successful_tests)) {
    test <- successful_tests[i, ]
    cat(sprintf("    %s: %d 个请求, %.2f秒, %.1fMB\n",
                test$config_name, test$request_count, test$processing_time, test$memory_used))
  }
}

# 显示失败的测试
failed_tests <- results[!results$success, ]
if (nrow(failed_tests) > 0) {
  cat("\n❌ 失败的测试：\n")
  for (i in 1:nrow(failed_tests)) {
    test <- failed_tests[i, ]
    cat(sprintf("    %s: %d 个请求失败 - %s\n",
                test$config_name, test$request_count, test$error_message))
  }
}

# 找出每个配置的最大成功规模
cat("\n🎯 各配置最大成功规模：\n")
for (config_name in unique(results$config_name)) {
  config_results <- results[results$config_name == config_name, ]
  successful_config <- config_results[config_results$success, ]
  
  if (nrow(successful_config) > 0) {
    max_successful <- max(successful_config$request_count)
    max_test <- successful_config[successful_config$request_count == max_successful, ][1, ]
    
    cat(sprintf("    %s: 最大 %d 个请求 (%.2f秒, %.1fMB)\n",
                config_name, max_successful, max_test$processing_time, max_test$memory_used))
  } else {
    cat(sprintf("    %s: 无成功测试\n", config_name))
  }
}

# 瓶颈分析
cat("\n🔍 瓶颈分析：\n")

# 检查是否受到 n_seq_max 限制
max_seq_limited <- any(results$request_count == results$n_seq_max & !results$success)
if (max_seq_limited) {
  cat("    📌 检测到 n_seq_max 限制 - 这是**模型配置限制**\n")
}

# 检查内存相关失败
memory_failures <- failed_tests[grepl("memory|内存|Memory", failed_tests$error_message, ignore.case = TRUE), ]
if (nrow(memory_failures) > 0) {
  cat("    📌 检测到内存相关失败 - 这是**电脑硬件限制**\n")
}

# 检查R相关失败
r_failures <- failed_tests[grepl("R|vector|allocation", failed_tests$error_message, ignore.case = TRUE), ]
if (nrow(r_failures) > 0) {
  cat("    📌 检测到R相关失败 - 这是**R Studio/R进程限制**\n")
}

# 检查其他失败
other_failures <- failed_tests[!grepl("memory|内存|Memory|R|vector|allocation", failed_tests$error_message, ignore.case = TRUE), ]
if (nrow(other_failures) > 0) {
  cat("    📌 检测到其他失败 - 可能是**并行生成函数限制**\n")
  cat("    错误信息：", paste(unique(other_failures$error_message), collapse = "; "), "\n")
}

# 性能趋势分析
if (nrow(successful_tests) > 0) {
  cat("\n📈 性能趋势：\n")
  
  # 计算平均每个请求的处理时间
  successful_tests$time_per_request <- successful_tests$processing_time / successful_tests$request_count
  
  # 按请求数量排序
  successful_tests <- successful_tests[order(successful_tests$request_count), ]
  
  cat("    每个请求平均处理时间：\n")
  for (i in 1:min(10, nrow(successful_tests))) {
    test <- successful_tests[i, ]
    cat(sprintf("      %d 个请求: %.3f秒/个\n", test$request_count, test$time_per_request))
  }
}

# 保存详细结果
write.csv(results, "parallel_limits_test_results.csv", row.names = FALSE)
cat("\n💾 详细结果已保存到 parallel_limits_test_results.csv\n")

# 清理
backend_free()
gc()
cat("\n🎉 大规模并行上限测试完成！\n")