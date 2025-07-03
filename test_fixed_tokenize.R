#!/usr/bin/env Rscript

# 测试修复后的tokenize函数
cat("=== Testing Fixed Tokenize Function ===\n\n")

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

# 3. 测试修复后的tokenize函数
cat("3. Testing fixed tokenize function with alignment improvements...\n")
tryCatch({
  cat("   Calling tokenize function...\n")
  
  # 测试简单的单字符
  result <- tokenize(model, "H", add_special = FALSE)
  
  cat("✅ Tokenize successful!\n")
  cat("   Result type:", typeof(result), "\n")
  cat("   Result class:", class(result), "\n")
  cat("   Result length:", length(result), "\n")
  cat("   Result values:", result, "\n\n")
  
  # 测试更长的文本
  cat("   Testing longer text...\n")
  result2 <- tokenize(model, "Hello world", add_special = FALSE)
  cat("✅ Longer text tokenize successful!\n")
  cat("   Result length:", length(result2), "\n")
  cat("   Result values:", result2, "\n\n")
  
}, error = function(e) {
  cat("❌ Tokenize failed:", conditionMessage(e), "\n")
  print(e)
})

cat("=== Test Complete ===\n") 