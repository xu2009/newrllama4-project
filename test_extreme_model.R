library(newrllama4)

# 测试12B极限模型
cat("=== 测试12B极限模型 ===\n")
cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"
model_url <- "https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q8_0.gguf"
model_filename <- "gemma-3-12b-it-Q8_0.gguf"
model_path <- file.path(cache_dir, model_filename)

cat("模型URL:", model_url, "\n")
cat("存储路径:", model_path, "\n")

# 1. 首先检查模型是否已存在
if (file.exists(model_path)) {
  cat("✓ 模型已存在，跳过下载\n")
  file_size <- file.info(model_path)$size / (1024^3)
  cat("文件大小:", round(file_size, 2), "GB\n")
} else {
  cat("模型不存在，开始下载...\n")
}

# 2. 测试内存估算（无论是否存在都可以测试）
cat("\n=== 内存估算测试 ===\n")

# 2.1 如果文件存在，直接估算
if (file.exists(model_path)) {
  cat("基于本地文件估算内存...\n")
  tryCatch({
    estimated_memory <- .Call("c_r_estimate_model_memory", model_path)
    cat("估算内存需求:", round(estimated_memory / (1024^3), 2), "GB\n")
    
    # 2.2 检查内存是否足够
    memory_available <- .Call("c_r_check_memory_available", estimated_memory)
    cat("内存是否足够:", memory_available, "\n")
    
    if (memory_available) {
      cat("✓ 内存足够，可以尝试加载模型\n")
    } else {
      cat("✗ 内存不足，无法加载模型\n")
      cat("这正是我们的内存保护机制发挥作用的时候！\n")
    }
    
  }, error = function(e) {
    cat("✗ 内存估算失败:", e$message, "\n")
  })
} else {
  cat("文件不存在，无法进行内存估算\n")
}

# 3. 尝试智能下载和加载
cat("\n=== 智能下载和加载测试 ===\n")
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
  
  cat("✓ 模型加载成功！\n")
  
  # 4. 如果成功加载，测试基本功能
  cat("\n=== 基本功能测试 ===\n")
  
  # 创建上下文
  context <- context_create(model, n_ctx = 512, n_threads = 4)
  cat("✓ 上下文创建成功\n")
  
  # 简单的tokenize测试
  text <- "Hello, how are you?"
  tokens <- tokenize(model, text, add_special = TRUE)
  cat("✓ tokenize成功，tokens数量:", length(tokens), "\n")
  
  # 简单的文本生成测试
  result <- generate(context, tokens, max_tokens = 20)
  cat("✓ 文本生成成功\n")
  cat("输入:", text, "\n")
  cat("输出:", result, "\n")
  
}, error = function(e) {
  cat("✗ 模型加载失败:", e$message, "\n")
  cat("错误类型分析:\n")
  
  # 检查是否是内存相关错误
  if (grepl("memory|Memory|内存", e$message)) {
    cat("- 这是内存相关错误，说明保护机制正常工作\n")
  }
  
  # 检查是否是下载相关错误
  if (grepl("download|网络|connection", e$message)) {
    cat("- 这是下载相关错误，可能是网络问题\n")
  }
  
  # 检查是否是文件完整性错误
  if (grepl("corrupt|integrity|完整性", e$message)) {
    cat("- 这是文件完整性错误，可能需要重新下载\n")
  }
})

# 5. 显示当前系统内存状态
cat("\n=== 系统内存状态 ===\n")
system("vm_stat | head -10")