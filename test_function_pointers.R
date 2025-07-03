#!/usr/bin/env Rscript

# 函数指针诊断测试
cat("=== Function Pointer Debug Test ===\n\n")

# 1. 加载包和初始化
cat("1. Loading and initializing...\n")
suppressPackageStartupMessages({
  library(newrllama4)
})
install_newrllama()

# 让我们在初始化之前和之后检查状态
cat("2. Before backend_init()...\n")
backend_init()
cat("✅ Backend initialized\n\n")

# 3. 加载模型
cat("3. Loading model...\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(
  model_path = model_path,
  n_gpu_layers = 0L,
  use_mmap = TRUE,
  use_mlock = FALSE
)
cat("✅ Model loaded\n\n")

# 4. 首先测试我们知道工作的函数
cat("4. Testing working functions...\n")
tryCatch({
  bos_token <- .Call("r_token_bos", model)
  cat("✅ r_token_bos successful: ", bos_token, "\n")
  
  eos_token <- .Call("r_token_eos", model)
  cat("✅ r_token_eos successful: ", eos_token, "\n")
  
}, error = function(e) {
  cat("❌ Known working functions failed:", conditionMessage(e), "\n")
})

# 5. 现在尝试一种非常保守的tokenize调用
cat("\n5. Conservative tokenize test...\n")

# 让我们尝试在函数中添加一些内存保护
tryCatch({
  cat("   Attempting tokenize with memory barriers...\n")
  
  # 强制刷新内存
  invisible(gc())
  
  # 使用最简单的参数
  result <- .Call("r_tokenize", model, as.character("H"), as.logical(FALSE))
  cat("✅ Conservative tokenize successful!\n")
  cat("   Result: ", result, "\n")
  
}, error = function(e) {
  cat("❌ Conservative tokenize failed: ", conditionMessage(e), "\n")
  
  # 让我们尝试捕获更详细的错误信息
  cat("Error details:\n")
  cat("   Class:", class(e), "\n")
  cat("   Message:", conditionMessage(e), "\n")
  traceback()
})

cat("\n=== Function Pointer Debug Complete ===\n") 