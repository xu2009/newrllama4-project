#!/usr/bin/env Rscript

# 基本的newrllama4测试脚本
cat("=== Basic newrllama4 Test ===\n\n")

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
    n_gpu_layers = 0L,     # CPU only for testing
    use_mmap = TRUE,
    use_mlock = FALSE
  )
  cat("✅ Model loaded successfully\n")
  cat("   Model type:", class(model), "\n\n")
}, error = function(e) {
  cat("❌ Model loading failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 6. 测试分词功能
cat("6. Testing tokenization...\n")
test_text <- "Hello world"
tryCatch({
  tokens <- tokenize(model, test_text, add_special = TRUE)
  cat("✅ Tokenization successful\n")
  cat("   Input text:", test_text, "\n")
  cat("   Tokens:", length(tokens), "tokens:", toString(head(tokens, 10)), "\n\n")
}, error = function(e) {
  cat("❌ Tokenization failed:", conditionMessage(e), "\n")
})

# 7. 创建上下文（使用更小的参数）
cat("7. Creating context with conservative parameters...\n")
tryCatch({
  context <- context_create(
    model = model,
    n_ctx = 512L,       # Smaller context length
    n_threads = 2L,     # Fewer threads
    n_seq_max = 1L      # Single sequence only
  )
  cat("✅ Context created successfully\n")
  cat("   Context type:", class(context), "\n\n")
}, error = function(e) {
  cat("❌ Context creation failed:", conditionMessage(e), "\n")
})

cat("=== Basic Test Summary ===\n")
cat("✅ Package loading: SUCCESS\n")
cat("✅ Backend installation: SUCCESS\n")
cat("✅ Backend initialization: SUCCESS\n")
cat("✅ Model loading: SUCCESS\n")
cat("? Tokenization: See above\n")
cat("? Context creation: See above\n")
cat("\n=== Basic test completed ===\n") 