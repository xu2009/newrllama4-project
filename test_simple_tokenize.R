#!/usr/bin/env Rscript

# 简化tokenize测试
cat("=== Simple Tokenize Test ===\n\n")

# 1. 加载包和初始化
cat("1. Loading and initializing...\n")
suppressPackageStartupMessages({
  if ("package:newrllama4" %in% search()) {
    detach("package:newrllama4", unload = TRUE, force = TRUE)
  }
  library(newrllama4)
})
install_newrllama()
backend_init()
cat("✅ Initialization complete\n\n")

# 2. 加载模型
cat("2. Loading model...\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(
  model_path = model_path,
  n_gpu_layers = 0L,
  use_mmap = TRUE,
  use_mlock = FALSE
)
cat("✅ Model loaded\n\n")

# 3. 测试简化的tokenize函数
cat("3. Testing simplified tokenize function...\n")
tryCatch({
  cat("   Calling tokenize_test...\n")
  
  # 使用R wrapper函数
  result <- tokenize_test(model)
  
  cat("✅ Simplified tokenize successful!\n")
  cat("   Result type:", typeof(result), "\n")
  cat("   Result class:", class(result), "\n")
  cat("   Result length:", length(result), "\n")
  cat("   Result values:", result, "\n\n")
  
}, error = function(e) {
  cat("❌ Simplified tokenize failed:", conditionMessage(e), "\n")
  print(e)
})

cat("=== Simple Tokenize Test Complete ===\n") 