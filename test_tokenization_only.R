#!/usr/bin/env Rscript

# 分词专用测试
cat("=== Tokenization Test ===\n\n")

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

# 3. 测试分词 - 先用很简单的文本
cat("3. Testing tokenization with simple text...\n")
test_text <- "Hello"
cat("   Input text:", test_text, "\n")

tryCatch({
  cat("   Calling tokenize function...\n")
  tokens <- tokenize(model, test_text, add_special = TRUE)
  cat("✅ Tokenization successful!\n")
  cat("   Number of tokens:", length(tokens), "\n")
  cat("   Tokens:", toString(tokens), "\n\n")
}, error = function(e) {
  cat("❌ Tokenization failed:", conditionMessage(e), "\n")
  cat("   This indicates a pointer or memory alignment issue\n")
  cat("   Let's try with more debugging...\n\n")
})

# 4. 如果失败，测试非常基础的东西
cat("4. Testing basic model properties...\n")
tryCatch({
  cat("   Model class:", class(model), "\n")
  cat("   Model pointer address:", format(model), "\n")
  cat("✅ Model pointer accessible\n\n")
}, error = function(e) {
  cat("❌ Model pointer issues:", conditionMessage(e), "\n")
})

# 5. 不要在脚本结束时访问模型，直接退出
cat("=== Test Complete ===\n")
cat("Exiting immediately to avoid cleanup issues...\n")
quit(save = "no", status = 0) 