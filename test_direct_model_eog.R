# ============================================================================
# 🔍 直接模型EOG token行为测试（绕过quick_llama缓存）
# ============================================================================
#
# 目的：直接使用底层API绕过quick_llama的缓存机制，
#       真正测试不同模型的EOG token处理差异
# ============================================================================

library(newrllama4)

# 测试模型配置
test_models <- list(
  "Gemma-3-1B" = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  "Llama-2-7B" = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf", 
  "Llama-3.2-3B" = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"
)

# 测试配置
test_prompt <- "Tell me a joke"
max_tokens <- 50
seed <- 12345

# 存储结果
results <- list()

cat("🔍 直接模型EOG token行为测试\n")
cat(paste0(rep("=", 50), collapse=""), "\n\n")

for (model_name in names(test_models)) {
  model_path <- test_models[[model_name]]
  
  cat("📋 测试模型:", model_name, "\n")
  cat("📁 路径:", basename(model_path), "\n")
  
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
    # 确保清理之前的状态
    backend_free()
    Sys.sleep(1)
    
    cat("🔄 直接加载模型...\n")
    
    # 直接使用底层API，启用GPU加速
    model <- model_load(model_path, n_gpu_layers = 999)  # 全部加载到GPU
    cat("✅ 模型加载成功 (GPU加速)\n")
    
    # 创建上下文
    ctx <- context_create(model, n_ctx = 2048)
    cat("✅ 上下文创建成功\n")
    
    # tokenize输入
    tokens <- tokenize(model, test_prompt, add_special = TRUE)
    cat("📝 Tokenized输入:", length(tokens), "tokens\n")
    
    # 直接调用生成函数（绕过所有清理逻辑）
    raw_output <- generate(ctx, tokens, 
                          max_tokens = max_tokens, 
                          temperature = 0.1, 
                          seed = seed)
    
    cat("✅ 生成成功\n")
    cat("📝 原始输出 (", nchar(raw_output), "字符):\n")
    cat("\"", raw_output, "\"\n")
    
    # 检查EOG tokens
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
      if (grepl(pattern, raw_output)) {
        matches <- gregexpr(pattern, raw_output)[[1]]
        if (matches[1] != -1) {
          found_eogs <- c(found_eogs, gsub("\\\\", "", pattern))
          cat("🔴 发现EOG token:", gsub("\\\\", "", pattern), "\n")
        }
      }
    }
    
    if (length(found_eogs) == 0) {
      cat("✅ 未发现EOG token泄漏\n")
    }
    
    # 记录详细结果
    results[[model_name]] <- list(
      status = "success",
      raw_output = raw_output,
      output_length = nchar(raw_output),
      eog_tokens_found = found_eogs,
      has_eog_leak = length(found_eogs) > 0,
      num_tokens_generated = max_tokens  # 实际可能更少
    )
    
    # 清理资源
    rm(model, ctx)
    backend_free()
    Sys.sleep(1)
    
  }, error = function(e) {
    cat("❌ 测试失败:", e$message, "\n")
    results[[model_name]] <- list(
      status = "error", 
      error = e$message
    )
    
    # 尝试清理
    tryCatch({
      backend_free()
    }, error = function(e2) {})
  })
  
  cat("\n", paste0(rep("-", 40), collapse=""), "\n\n")
}

# ============================================================================
# 详细分析
# ============================================================================

cat("📊 详细测试结果分析\n")
cat(paste0(rep("=", 50), collapse=""), "\n\n")

successful_tests <- 0
models_with_leaks <- 0

for (model_name in names(results)) {
  result <- results[[model_name]]
  
  cat("🔹", model_name, ":\n")
  
  if (result$status == "success") {
    successful_tests <- successful_tests + 1
    
    cat("  📏 输出长度:", result$output_length, "字符\n")
    
    if (result$has_eog_leak) {
      models_with_leaks <- models_with_leaks + 1
      cat("  🔴 EOG泄漏: 是\n")
      cat("  🚨 泄漏tokens:", paste(result$eog_tokens_found, collapse=", "), "\n")
    } else {
      cat("  ✅ EOG泄漏: 否\n")
    }
    
    # 显示输出的前100字符用于对比
    preview <- if(nchar(result$raw_output) > 100) {
      paste0(substr(result$raw_output, 1, 100), "...")
    } else {
      result$raw_output
    }
    cat("  📖 输出预览:", preview, "\n")
    
  } else {
    cat("  ❌ 状态:", result$status, "\n")
    if (!is.null(result$error)) {
      cat("  📝 错误:", result$error, "\n")
    }
  }
  
  cat("\n")
}

# ============================================================================
# 最终结论
# ============================================================================

cat("🎯 最终分析结论\n")
cat(paste0(rep("=", 50), collapse=""), "\n")

cat("📈 成功测试:", successful_tests, "/", length(test_models), "个模型\n")
cat("🔴 发现EOG泄漏:", models_with_leaks, "/", successful_tests, "个模型\n")

if (successful_tests == 0) {
  cat("\n❌ 所有测试都失败了，需要检查模型文件或实现。\n")
  
} else if (models_with_leaks == 0) {
  cat("\n✅ 太好了！所有模型都没有EOG token泄漏。\n")
  cat("💭 这可能意味着:\n")
  cat("   1. 当前的修复已经有效\n")
  cat("   2. 或者问题出现在特定的使用场景中\n")
  cat("   3. 需要更复杂的prompt来触发EOG泄漏\n")
  
} else if (models_with_leaks == successful_tests) {
  cat("\n🔴 所有模型都有EOG泄漏！这表明问题在我们的实现中。\n")
  
} else {
  cat("\n📊 混合结果：部分模型有泄漏，部分没有。\n") 
  cat("💡 这证实了EOG token处理确实是模型特异的！\n")
  
  cat("\n🔍 模型对比:\n")
  for (model_name in names(results)) {
    result <- results[[model_name]]
    if (result$status == "success") {
      icon <- if (result$has_eog_leak) "🔴" else "✅"
      status <- if (result$has_eog_leak) "有泄漏" else "无泄漏"
      cat(sprintf("  %s %s: %s\n", icon, model_name, status))
    }
  }
}

cat("\n=== 直接模型测试完成 ===\n")