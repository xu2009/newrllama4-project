library(newrllama4)

# 安装后端库
install_newrllama()

# 测试1：1GB模型下载和加载
cat("=== 测试1：1GB模型 (gemma-3-1b-it.Q8_0.gguf) ===\n")
model_url_1gb <- "https://huggingface.co/MaziyarPanahi/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it.Q8_0.gguf"
cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"

# 确保目录存在
if (!dir.exists(cache_dir)) {
  dir.create(cache_dir, recursive = TRUE)
}

# 测试1GB模型
cat("开始下载和加载1GB模型...\n")
result_1gb <- tryCatch({
  model_1gb <- model_load(model_url_1gb, cache_dir = cache_dir, show_progress = TRUE, verify_integrity = TRUE, check_memory = TRUE)
  cat("✓ 1GB模型加载成功\n")
  model_1gb
}, error = function(e) {
  cat("✗ 1GB模型加载失败:", e$message, "\n")
  NULL
})

# 测试生成（如果模型加载成功）
if (!is.null(result_1gb)) {
  cat("测试文本生成...\n")
  tryCatch({
    result <- generate(result_1gb, "Hello, how are you?", max_tokens = 50)
    cat("✓ 文本生成成功:", result, "\n")
  }, error = function(e) {
    cat("✗ 文本生成失败:", e$message, "\n")
  })
}

# 测试2：8GB模型下载和加载
cat("\n=== 测试2：8GB模型 (DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf) ===\n")
model_url_8gb <- "https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF/resolve/main/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf"

cat("开始下载和加载8GB模型...\n")
result_8gb <- tryCatch({
  model_8gb <- model_load(model_url_8gb, cache_dir = cache_dir, show_progress = TRUE, verify_integrity = TRUE, check_memory = TRUE)
  cat("✓ 8GB模型加载成功\n")
  model_8gb
}, error = function(e) {
  cat("✗ 8GB模型加载失败:", e$message, "\n")
  NULL
})

# 测试生成（如果模型加载成功）
if (!is.null(result_8gb)) {
  cat("测试文本生成...\n")
  tryCatch({
    result <- generate(result_8gb, "Hello, how are you?", max_tokens = 50)
    cat("✓ 文本生成成功:", result, "\n")
  }, error = function(e) {
    cat("✗ 文本生成失败:", e$message, "\n")
  })
}

# 检查缓存目录
cat("\n=== 检查缓存目录 ===\n")
if (dir.exists(cache_dir)) {
  files <- list.files(cache_dir, recursive = TRUE, full.names = TRUE)
  cat("缓存目录中的文件:\n")
  for (file in files) {
    file_size <- file.info(file)$size / (1024^3)  # GB
    cat(sprintf("  %s (%.2f GB)\n", basename(file), file_size))
  }
} else {
  cat("缓存目录不存在\n")
}

cat("\n=== 测试完成 ===\n")