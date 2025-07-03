#!/usr/bin/env Rscript

# 内存对齐诊断测试
cat("=== Memory Alignment Debug Test ===\n\n")

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
cat("✅ Model loaded\n")
cat("   Model object class:", class(model), "\n")
cat("   Model object type:", typeof(model), "\n\n")

# 3. 检查模型指针地址
cat("3. Checking model pointer alignment...\n")
# 在R中，我们无法直接检查指针地址，但我们可以测试一些基本操作
tryCatch({
  cat("   Testing model pointer validity...\n")
  # 获取模型对象的属性
  model_attrs <- attributes(model)
  cat("   Model attributes:", names(model_attrs), "\n")
  cat("✅ Model pointer seems valid\n\n")
}, error = function(e) {
  cat("❌ Model pointer validation failed:", conditionMessage(e), "\n")
})

# 4. 测试极简单的tokenization调用
cat("4. Testing ultra-simple tokenization...\n")
cat("   Attempting to tokenize single character...\n")

# 尝试最简单可能的情况
tryCatch({
  # 使用最简单的输入
  simple_text <- "H"  # 单个字符
  cat("   Input text: '", simple_text, "'\n", sep="")
  cat("   Calling tokenize with minimal parameters...\n")
  
  # 直接调用底层函数，避免任何额外的处理
  result <- tokenize(model, simple_text, add_special = FALSE)
  
  cat("✅ Tokenization successful!\n")
  cat("   Result type:", typeof(result), "\n")
  cat("   Result class:", class(result), "\n")
  cat("   Result length:", length(result), "\n")
  cat("   Result values:", result, "\n\n")
  
}, error = function(e) {
  cat("❌ Tokenization failed with error:\n")
  cat("   Error message:", conditionMessage(e), "\n")
  cat("   Error class:", class(e), "\n\n")
  
  # 尝试捕获更多信息
  cat("Additional error details:\n")
  print(e)
})

# 5. 尝试不同的文本输入
cat("5. Testing different text inputs...\n")
test_cases <- c("a", "1", " ", "\n")

for (i in seq_along(test_cases)) {
  text <- test_cases[i]
  cat("   Test case", i, ": '", text, "' (ASCII:", utf8ToInt(text), ")\n")
  
  tryCatch({
    result <- tokenize(model, text, add_special = FALSE)
    cat("     ✅ Success: tokens =", result, "\n")
  }, error = function(e) {
    cat("     ❌ Failed:", conditionMessage(e), "\n")
  })
}

cat("\n=== Alignment Debug Complete ===\n") 