#!/usr/bin/env Rscript

# 直接调用测试 - 绕过函数指针
cat("=== Direct Call Test ===\n\n")

# 1. 加载包和初始化
cat("1. Loading and initializing...\n")
suppressPackageStartupMessages({
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

# 3. 测试使用.C或.Call直接调用
cat("3. Testing direct C function call...\n")

# 首先检查是否可以调用一个不需要参数的简单函数
tryCatch({
  cat("   Trying to call simple token function...\n")
  # 尝试调用一个简单的token函数，看看是否有同样的对齐问题
  bos_token <- .Call("r_token_bos", model)
  cat("✅ Direct call successful! BOS token:", bos_token, "\n\n")
}, error = function(e) {
  cat("❌ Direct call failed:", conditionMessage(e), "\n\n")
})

# 4. 测试不同的调用方式
cat("4. Testing tokenization with different approach...\n")

# 让我们尝试通过一个不同的路径调用tokenize
tryCatch({
  cat("   Trying alternative tokenize call...\n")
  # 也许问题在于参数的传递方式
  result <- .Call("r_tokenize", model, "H", FALSE)
  cat("✅ Alternative tokenize successful!\n")
  cat("   Result:", result, "\n\n")
}, error = function(e) {
  cat("❌ Alternative tokenize failed:", conditionMessage(e), "\n\n")
})

cat("=== Direct Call Test Complete ===\n") 