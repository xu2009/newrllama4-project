library(newrllama4)

# 测试完整的工作流程
cat("=== 测试完整工作流程 ===\n")
cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"
model_1gb_path <- file.path(cache_dir, "gemma-3-1b-it.Q8_0.gguf")

if (file.exists(model_1gb_path)) {
  cat("✓ 1GB模型已存在:", model_1gb_path, "\n")
  
  # 1. 测试内存估算
  cat("\n1. 测试内存估算...\n")
  estimated_memory <- .Call("c_r_estimate_model_memory", model_1gb_path)
  cat("估算内存:", round(estimated_memory / (1024^3), 2), "GB\n")
  
  # 2. 测试内存检查
  memory_available <- .Call("c_r_check_memory_available", estimated_memory)
  cat("内存是否足够:", memory_available, "\n")
  
  # 3. 测试模型加载
  cat("\n2. 测试模型加载...\n")
  tryCatch({
    model <- model_load(model_1gb_path, check_memory = TRUE, verify_integrity = TRUE)
    cat("✓ 模型加载成功\n")
    
    # 4. 创建上下文
    cat("\n3. 创建推理上下文...\n")
    context <- context_create(model, n_ctx = 512, n_threads = 4)
    cat("✓ 上下文创建成功\n")
    
    # 5. 测试tokenize
    cat("\n4. 测试tokenize...\n")
    text <- "Hello, how are you?"
    tokens <- tokenize(model, text, add_special = TRUE)
    cat("✓ tokenize成功，tokens:", length(tokens), "个\n")
    
    # 6. 测试文本生成
    cat("\n5. 测试文本生成...\n")
    result <- generate(context, tokens, max_tokens = 30)
    cat("✓ 生成成功\n")
    cat("输入:", text, "\n")
    cat("输出:", result, "\n")
    
    cat("\n=== 1GB模型测试完成 ===\n")
    
  }, error = function(e) {
    cat("✗ 错误:", e$message, "\n")
  })
} else {
  cat("✗ 1GB模型不存在\n")
}

# 测试8GB模型（如果存在）
model_8gb_path <- file.path(cache_dir, "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf")
if (file.exists(model_8gb_path)) {
  cat("\n=== 测试8GB模型 ===\n")
  
  # 估算内存
  estimated_memory_8gb <- .Call("c_r_estimate_model_memory", model_8gb_path)
  cat("8GB模型估算内存:", round(estimated_memory_8gb / (1024^3), 2), "GB\n")
  
  # 检查内存
  memory_available_8gb <- .Call("c_r_check_memory_available", estimated_memory_8gb)
  cat("8GB模型内存是否足够:", memory_available_8gb, "\n")
  
  if (memory_available_8gb) {
    cat("✓ 内存足够，可以尝试加载8GB模型\n")
  } else {
    cat("✗ 内存不足，无法加载8GB模型\n")
  }
} else {
  cat("\n8GB模型不存在，跳过测试\n")
}