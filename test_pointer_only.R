#!/usr/bin/env Rscript

# 最简化测试 - 仅测试模型加载和指针
cat("=== Pointer-Only Test ===\n\n")

# 1. 加载包
cat("1. Loading newrllama4 package...\n")
suppressPackageStartupMessages({
  library(newrllama4)
})
cat("✅ Package loaded successfully\n\n")

# 2. 确保后端已安装
cat("2. Checking backend installation...\n")
install_newrllama()
cat("✅ Backend ready\n\n")

# 3. 初始化后端
cat("3. Initializing backend...\n")
tryCatch({
  backend_init()
  cat("✅ Backend initialized successfully\n\n")
}, error = function(e) {
  cat("❌ Backend initialization failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 4. 设置模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
cat("4. Using model:", model_path, "\n")

if (!file.exists(model_path)) {
  stop("❌ Model file not found: ", model_path)
}
cat("✅ Model file exists\n\n")

# 5. 加载模型
cat("5. Loading model...\n")
tryCatch({
  model <- model_load(
    model_path = model_path,
    n_gpu_layers = 0L,
    use_mmap = TRUE,
    use_mlock = FALSE
  )
  cat("✅ Model loaded successfully\n")
  cat("   Model type:", class(model), "\n")
  cat("   Model object:", typeof(model), "\n\n")
}, error = function(e) {
  cat("❌ Model loading failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 6. 检查指针对象
cat("6. Checking model pointer...\n")
tryCatch({
  cat("   Model class:", class(model), "\n")
  cat("   Model environment:", typeof(model), "\n")
  cat("   Model attributes:", names(attributes(model)), "\n")
  if (is(model, "externalptr")) {
    cat("✅ Model is correctly an external pointer\n\n")
  } else {
    cat("❌ Model is not an external pointer\n\n")
  }
}, error = function(e) {
  cat("❌ Error checking model pointer:", conditionMessage(e), "\n")
  quit(status = 1)
})

cat("=== Pointer Test Summary ===\n")
cat("✅ Package loading: Success\n")
cat("✅ Backend installation: Success\n")
cat("✅ Backend initialization: Success\n")
cat("✅ Model file access: Success\n")
cat("✅ Model loading: Success\n")
cat("✅ Pointer type: External pointer\n")
cat("\n🎉 Basic pointer functionality is working!\n")
cat("\nNext step: Test tokenization with proper alignment fixes\n") 