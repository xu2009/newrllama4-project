library(newrllama4)

# 测试12B模型下载和内存保护
cat("=== 测试12B模型下载和内存保护 ===\n")

cache_dir <- "/Users/yaoshengleo/Desktop/gguf模型"
model_url <- "https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q8_0.gguf"
model_filename <- "gemma-3-12b-it-Q8_0.gguf"
model_path <- file.path(cache_dir, model_filename)

cat("模型URL:", model_url, "\n")
cat("目标路径:", model_path, "\n")

# 检查文件是否存在
if (file.exists(model_path)) {
  cat("✓ 模型文件已存在，删除重新下载\n")
  file.remove(model_path)
}

# 创建目录
if (!dir.exists(cache_dir)) {
  dir.create(cache_dir, recursive = TRUE)
}

# 开始下载测试
cat("\n=== 开始下载测试 ===\n")
cat("这将是一个持续的下载过程，请耐心等待...\n")

start_time <- Sys.time()

tryCatch({
  # 直接调用下载函数
  result <- .Call("c_r_download_model", 
                 model_url, 
                 model_path, 
                 TRUE)  # 显示进度
  
  end_time <- Sys.time()
  download_time <- end_time - start_time
  
  cat("\n✓ 下载完成！\n")
  cat("下载时间:", round(as.numeric(download_time), 2), "秒\n")
  
  # 检查文件大小
  if (file.exists(model_path)) {
    file_size <- file.info(model_path)$size / (1024^3)
    cat("文件大小:", round(file_size, 2), "GB\n")
    
    # 测试内存估算
    cat("\n=== 测试内存估算 ===\n")
    estimated_memory <- .Call("c_r_estimate_model_memory", model_path)
    cat("估算内存需求:", round(estimated_memory / (1024^3), 2), "GB\n")
    
    # 测试内存检查
    memory_available <- .Call("c_r_check_memory_available", estimated_memory)
    cat("内存是否足够:", memory_available, "\n")
    
    if (memory_available) {
      cat("✓ 内存足够，可以尝试加载模型\n")
      
      # 尝试加载模型
      cat("\n=== 尝试加载模型 ===\n")
      tryCatch({
        model <- model_load(model_path, 
                           check_memory = TRUE, 
                           verify_integrity = TRUE)
        cat("✓ 模型加载成功！\n")
        
        # 测试基本功能
        cat("\n=== 测试基本功能 ===\n")
        context <- context_create(model, n_ctx = 512, n_threads = 4)
        cat("✓ 上下文创建成功\n")
        
        text <- "Hello, how are you?"
        tokens <- tokenize(model, text, add_special = TRUE)
        cat("✓ tokenize成功，tokens:", length(tokens), "\n")
        
        result <- generate(context, tokens, max_tokens = 20)
        cat("✓ 文本生成成功\n")
        cat("输入:", text, "\n")
        cat("输出:", result, "\n")
        
      }, error = function(e) {
        cat("✗ 模型加载失败:", e$message, "\n")
      })
      
    } else {
      cat("✗ 内存不足，无法加载模型\n")
      cat("这证明了内存保护机制正在工作！\n")
    }
  }
  
}, error = function(e) {
  end_time <- Sys.time()
  download_time <- end_time - start_time
  
  cat("\n✗ 下载失败:", e$message, "\n")
  cat("下载时间:", round(as.numeric(download_time), 2), "秒\n")
  
  # 检查是否有部分下载的文件
  if (file.exists(model_path)) {
    file_size <- file.info(model_path)$size / (1024^3)
    cat("部分下载文件大小:", round(file_size, 2), "GB\n")
  }
})

# 显示系统内存状态
cat("\n=== 系统内存状态 ===\n")
system("vm_stat | head -6")