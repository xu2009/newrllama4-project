# ============================================================================
# 🔍 多模型EOG token行为对比测试
# ============================================================================
#
# 目的：验证不同模型的EOG token处理是否符合llama.cpp的单token假设
# 
# 测试模型：
# 1. Gemma-3-1B: Google的模型，使用不同的tokenizer
# 2. Llama-2-7B: 较老的Llama模型，可能使用传统单token EOG
# 3. Llama-3.2-3B: 已知有multi-token EOG问题的模型
# ============================================================================

library(newrllama4)

# 测试配置
test_models <- list(
  "Gemma-3-1B" = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  "Llama-2-7B" = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf", 
  "Llama-3.2-3B" = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"
)

# 测试prompt - 简单的问候，容易触发EOG
test_prompt <- "Hello, how are you today?"

# 存储结果
results <- list()

cat("🔍 开始多模型EOG token行为对比测试\n")
cat(paste0(rep("=", 60), collapse=""), "\n\n")

# 测试每个模型
for (model_name in names(test_models)) {
  model_path <- test_models[[model_name]]
  
  cat("📋 测试模型:", model_name, "\n")
  cat("📁 路径:", model_path, "\n")
  
  # 检查文件是否存在
  if (!file.exists(model_path)) {
    cat("❌ 文件不存在，跳过\n\n")
    results[[model_name]] <- list(
      status = "file_not_found",
      error = "Model file does not exist"
    )
    next
  }
  
  tryCatch({
    # 重置环境
    quick_llama_reset()
    
    cat("🔄 加载模型...\n")
    
    # 使用quick_llama进行测试
    result <- quick_llama(
      prompt = test_prompt,
      model_url = model_path,
      max_tokens = 50,
      temperature = 0.1,  # 低温度确保一致性
      seed = 12345,       # 固定种子
      auto_format = FALSE # 关闭自动清理，观察原始输出
    )
    
    cat("✅ 生成成功\n")
    cat("📝 原始输出:\n")
    cat("\"", result, "\"\n")
    cat("📏 输出长度:", nchar(result), "字符\n")
    
    # 检查是否包含已知的EOG tokens
    eog_patterns <- c(
      "<\\|eot_id\\|>", 
      "<\\|end_header_id\\|>", 
      "<\\|start_header_id\\|>",
      "<\\|im_end\\|>",
      "<\\|im_start\\|>", 
      "<end_of_turn>",
      "<\\|endoftext\\|>",
      "</s>"
    )
    
    found_eogs <- c()
    for (pattern in eog_patterns) {
      if (grepl(pattern, result)) {
        matches <- gregexpr(pattern, result)[[1]]
        if (matches[1] != -1) {
          found_eogs <- c(found_eogs, pattern)
          cat("🔴 发现EOG token:", pattern, "在位置:", paste(matches, collapse=", "), "\n")
        }
      }
    }
    
    if (length(found_eogs) == 0) {
      cat("✅ 未发现已知EOG token泄漏\n")
    }
    
    # 记录结果
    results[[model_name]] <- list(
      status = "success",
      output = result,
      output_length = nchar(result),
      eog_tokens_found = found_eogs,
      has_eog_leak = length(found_eogs) > 0
    )
    
    # 清理资源
    quick_llama_reset()
    
  }, error = function(e) {
    cat("❌ 测试失败:", e$message, "\n")
    results[[model_name]] <- list(
      status = "error", 
      error = e$message
    )
    
    # 尝试清理
    tryCatch(quick_llama_reset(), error = function(e2) {})
  })
  
  cat("\n", paste0(rep("-", 40), collapse=""), "\n\n")
}

# ============================================================================
# 结果分析和总结
# ============================================================================

cat("📊 测试结果总结\n")
cat(paste0(rep("=", 60), collapse=""), "\n\n")

successful_tests <- 0
models_with_eog_leaks <- 0

for (model_name in names(results)) {
  result <- results[[model_name]]
  
  cat("🔸", model_name, ":\n")
  
  if (result$status == "success") {
    successful_tests <- successful_tests + 1
    
    if (result$has_eog_leak) {
      models_with_eog_leaks <- models_with_eog_leaks + 1
      cat("  ❌ 状态: EOG token泄漏\n")
      cat("  🔴 泄漏的tokens:", paste(result$eog_tokens_found, collapse=", "), "\n")
    } else {
      cat("  ✅ 状态: 无EOG token泄漏\n")
    }
    
    cat("  📏 输出长度:", result$output_length, "字符\n")
    
  } else if (result$status == "file_not_found") {
    cat("  ⚠️  状态: 文件未找到\n")
    
  } else {
    cat("  ❌ 状态: 测试失败\n")
    cat("  📝 错误:", result$error, "\n")
  }
  
  cat("\n")
}

# ============================================================================
# 关键结论
# ============================================================================

cat("🎯 关键结论\n")
cat(paste0(rep("=", 60), collapse=""), "\n")

cat("📈 成功测试的模型数量:", successful_tests, "/", length(test_models), "\n")
cat("🔴 有EOG泄漏的模型数量:", models_with_eog_leaks, "/", successful_tests, "\n")

if (models_with_eog_leaks == 0) {
  cat("\n✅ 太好了！所有测试的模型都没有EOG token泄漏问题。\n")
  cat("💡 这说明问题可能特定于某些Llama-3.2版本或特定的tokenizer配置。\n")
  
} else if (models_with_eog_leaks == successful_tests) {
  cat("\n🔴 所有模型都有EOG token泄漏问题！\n") 
  cat("💡 这说明问题可能在我们的实现中，而不是模型特异的。\n")
  
} else {
  cat("\n📊 部分模型有EOG泄漏问题，部分没有。\n")
  cat("💡 这证实了EOG token处理确实是模型特异的！\n")
  
  cat("\n🔍 详细分析:\n")
  for (model_name in names(results)) {
    result <- results[[model_name]]
    if (result$status == "success") {
      status_icon <- if (result$has_eog_leak) "❌" else "✅"
      cat(sprintf("  %s %s: %s\n", status_icon, model_name, 
                  if (result$has_eog_leak) "有泄漏" else "无泄漏"))
    }
  }
}

cat("\n🚀 基于这些结果，我们可以确定最佳的修复策略。\n")

cat("\n=== 多模型EOG测试完成 ===\n")