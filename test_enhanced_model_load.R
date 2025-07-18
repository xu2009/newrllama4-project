library(newrllama4)

# 检查现有的1GB模型
cat("=== 检查现有模型 ===\n")
cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"
model_1gb_path <- file.path(cache_dir, "gemma-3-1b-it.Q8_0.gguf")

if (file.exists(model_1gb_path)) {
  cat("✓ 1GB模型已存在:", model_1gb_path, "\n")
  
  # 测试内存估算
  cat("估算内存需求...\n")
  estimated_memory <- .Call("c_r_estimate_model_memory", model_1gb_path)
  cat("估算内存:", round(estimated_memory / (1024^3), 2), "GB\n")
  
  # 测试内存检查
  memory_available <- .Call("c_r_check_memory_available", estimated_memory)
  cat("内存是否足够:", memory_available, "\n")
  
  # 测试增强的模型加载
  cat("\n测试增强的模型加载...\n")
  tryCatch({
    model <- model_load(model_1gb_path, check_memory = TRUE, verify_integrity = TRUE)
    cat("✓ 模型加载成功\n")
    
    # 测试简单的文本生成
    cat("测试文本生成...\n")
    result <- generate(model, "Hello, how are you?", max_tokens = 20)
    cat("✓ 生成结果:", result, "\n")
    
  }, error = function(e) {
    cat("✗ 模型加载失败:", e$message, "\n")
  })
} else {
  cat("✗ 1GB模型不存在，需要重新下载\n")
}