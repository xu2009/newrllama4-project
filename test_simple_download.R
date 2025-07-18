library(newrllama4)

# 简单测试下载功能
cat("=== 简单测试下载功能 ===\n")

# 测试小文件下载
cat("测试小文件下载...\n")
tryCatch({
  result <- .Call("c_r_download_model", 
                 "https://httpbin.org/robots.txt", 
                 "/tmp/test_download.txt", 
                 TRUE)
  cat("✓ 小文件下载成功\n")
  if (file.exists("/tmp/test_download.txt")) {
    cat("文件内容:\n")
    cat(readLines("/tmp/test_download.txt", n = 3), sep = "\n")
    file.remove("/tmp/test_download.txt")
  }
}, error = function(e) {
  cat("✗ 小文件下载失败:", e$message, "\n")
})

# 测试内存估算函数
cat("\n=== 测试内存估算函数 ===\n")
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf"
if (file.exists(model_path)) {
  tryCatch({
    memory_est <- .Call("c_r_estimate_model_memory", model_path)
    cat("✓ 内存估算成功:", round(memory_est / (1024^3), 2), "GB\n")
  }, error = function(e) {
    cat("✗ 内存估算失败:", e$message, "\n")
  })
} else {
  cat("1GB测试模型不存在\n")
}

# 测试解析model路径
cat("\n=== 测试解析模型路径 ===\n")
tryCatch({
  resolved_path <- .Call("c_r_resolve_model", "https://httpbin.org/robots.txt")
  cat("✓ 路径解析成功:", resolved_path, "\n")
}, error = function(e) {
  cat("✗ 路径解析失败:", e$message, "\n")
})

cat("\n=== 测试完成 ===\n")