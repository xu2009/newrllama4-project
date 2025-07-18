library(newrllama4)

# 测试直接加载12B模型（绕过内存保护）
cat("=== 测试直接加载12B模型（无保护） ===\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-Q8_0.gguf"

cat("模型路径:", model_path, "\n")
cat("文件大小:", round(file.info(model_path)$size / (1024^3), 2), "GB\n")

# 显示加载前的内存状态
cat("\n=== 加载前的系统内存 ===\n")
system("vm_stat | head -6")

cat("\n=== 开始直接加载测试 ===\n")
cat("警告：这可能导致系统崩溃或R会话终止！\n")
cat("测试目的：验证无保护机制时的行为\n\n")

start_time <- Sys.time()

tryCatch({
  # 方法1：直接使用model_load但禁用保护
  cat("方法1：禁用内存检查的model_load...\n")
  model1 <- model_load(model_path, 
                      check_memory = FALSE,  # 禁用内存检查
                      verify_integrity = FALSE,  # 禁用完整性检查
                      n_gpu_layers = 0L,
                      use_mmap = TRUE)
  
  end_time <- Sys.time()
  load_time <- as.numeric(end_time - start_time)
  
  cat("✓ 方法1成功！加载时间:", round(load_time, 2), "秒\n")
  
  # 显示加载后的内存状态
  cat("\n=== 加载后的系统内存 ===\n")
  system("vm_stat | head -6")
  
  # 测试基本功能
  cat("\n=== 测试基本功能 ===\n")
  context <- context_create(model1, n_ctx = 512, n_threads = 2)
  cat("✓ 上下文创建成功\n")
  
  text <- "Hello"
  tokens <- tokenize(model1, text, add_special = TRUE)
  cat("✓ tokenize成功，tokens:", length(tokens), "\n")
  
  result <- generate(context, tokens, max_tokens = 10)
  cat("✓ 文本生成成功\n")
  cat("输入:", text, "\n")
  cat("输出:", result, "\n")
  
}, error = function(e) {
  end_time <- Sys.time()
  load_time <- as.numeric(end_time - start_time)
  
  cat("✗ 方法1失败，耗时:", round(load_time, 2), "秒\n")
  cat("错误信息:", e$message, "\n")
  
  # 分析错误类型
  if (grepl("memory|Memory|内存|out of memory|OOM", e$message, ignore.case = TRUE)) {
    cat("→ 内存相关错误\n")
  } else if (grepl("mmap|mapping|文件映射", e$message, ignore.case = TRUE)) {
    cat("→ 内存映射相关错误\n")
  } else if (grepl("allocation|分配", e$message, ignore.case = TRUE)) {
    cat("→ 内存分配相关错误\n")
  } else {
    cat("→ 其他类型错误\n")
  }
})

# 尝试方法2：直接调用底层C函数
cat("\n=== 方法2：直接调用底层C函数 ===\n")
tryCatch({
  start_time2 <- Sys.time()
  
  # 直接调用不安全的加载函数
  model2 <- .Call("c_r_model_load", 
                  model_path, 
                  0L,    # n_gpu_layers
                  TRUE,  # use_mmap
                  FALSE) # use_mlock
  
  end_time2 <- Sys.time()
  load_time2 <- as.numeric(end_time2 - start_time2)
  
  cat("✓ 方法2成功！加载时间:", round(load_time2, 2), "秒\n")
  
  # 显示最终内存状态
  cat("\n=== 最终系统内存 ===\n")
  system("vm_stat | head -6")
  
}, error = function(e) {
  end_time2 <- Sys.time()
  load_time2 <- as.numeric(end_time2 - start_time2)
  
  cat("✗ 方法2失败，耗时:", round(load_time2, 2), "秒\n")
  cat("错误信息:", e$message, "\n")
})

# 显示进程内存使用情况
cat("\n=== R进程内存使用 ===\n")
tryCatch({
  # 获取当前R进程的内存使用
  gc_info <- gc()
  cat("R内存使用情况:\n")
  print(gc_info)
  
  # 使用system命令查看进程内存
  pid <- Sys.getpid()
  cat("\nR进程ID:", pid, "\n")
  system(paste0("ps -o pid,rss,vsz,pmem,comm -p ", pid))
  
}, error = function(e) {
  cat("获取内存信息失败:", e$message, "\n")
})

cat("\n=== 测试完成 ===\n")