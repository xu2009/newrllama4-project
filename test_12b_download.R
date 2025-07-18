library(newrllama4)

# 测试12B模型下载（修复后的版本）
cat("=== 测试12B模型下载和内存保护 ===\n")

cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"
model_url <- "https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q8_0.gguf"
model_filename <- "gemma-3-12b-it-Q8_0.gguf"
model_path <- file.path(cache_dir, model_filename)

cat("模型URL:", model_url, "\n")
cat("目标路径:", model_path, "\n")

# 首先检查后端是否加载成功
tryCatch({
  api_info <- .Call("c_newrllama_api_init", NULL)
  cat("✓ newrllama后端加载成功\n")
}, error = function(e) {
  cat("✗ newrllama后端加载失败:", e$message, "\n")
  stop("无法继续测试")
})

# 测试下载功能是否可用
cat("\n=== 测试下载功能是否可用 ===\n")
tryCatch({
  # 先用一个小文件测试下载功能
  test_result <- .Call("c_r_download_model", 
                      "https://httpbin.org/robots.txt", 
                      "/tmp/test_download.txt", 
                      TRUE)
  cat("✓ 下载功能测试成功\n")
  file.remove("/tmp/test_download.txt")
}, error = function(e) {
  cat("✗ 下载功能测试失败:", e$message, "\n")
  stop("下载功能不可用")
})

# 开始12B模型下载和测试
cat("\n=== 开始12B模型智能下载 ===\n")
tryCatch({
  # 使用增强的model_load函数
  model <- model_load(
    model_url,
    cache_dir = cache_dir,
    check_memory = TRUE,
    verify_integrity = TRUE,
    show_progress = TRUE,
    n_gpu_layers = 0L,
    use_mmap = TRUE,
    use_mlock = FALSE
  )
  
  cat("✓ 12B模型加载成功！\n")
  
  # 测试基本功能
  cat("\n=== 测试基本功能 ===\n")
  context <- context_create(model, n_ctx = 512, n_threads = 4)
  cat("✓ 上下文创建成功\n")
  
  # 简单测试
  text <- "Hello, how are you?"
  tokens <- tokenize(model, text, add_special = TRUE)
  cat("✓ tokenize成功，tokens:", length(tokens), "\n")
  
  result <- generate(context, tokens, max_tokens = 20)
  cat("✓ 文本生成成功\n")
  cat("输入:", text, "\n")
  cat("输出:", result, "\n")
  
}, error = function(e) {
  cat("✗ 模型处理失败:", e$message, "\n")
  
  # 分析错误类型
  if (grepl("memory|Memory|内存", e$message)) {
    cat("→ 这是内存相关错误，内存保护机制正常工作\n")
  } else if (grepl("download|网络|connection", e$message)) {
    cat("→ 这是下载相关错误，可能是网络问题\n")
  } else {
    cat("→ 其他错误类型\n")
  }
  
  # 检查文件是否部分下载
  if (file.exists(model_path)) {
    file_size <- file.info(model_path)$size / (1024^3)
    cat("部分下载的文件大小:", round(file_size, 2), "GB\n")
    
    # 如果文件不完整，尝试内存估算
    if (file_size > 1) {  # 至少有1GB下载了
      cat("尝试对部分文件进行内存估算...\n")
      tryCatch({
        estimated_memory <- .Call("c_r_estimate_model_memory", model_path)
        cat("部分文件估算内存:", round(estimated_memory / (1024^3), 2), "GB\n")
      }, error = function(e2) {
        cat("部分文件内存估算失败:", e2$message, "\n")
      })
    }
  }
})

# 显示系统内存状态
cat("\n=== 系统内存状态 ===\n")
system("vm_stat | head -6")