library(newrllama4)

# 测试内存检查函数是否可用
cat("=== 测试内存检查函数 ===\n")

# 检查函数是否存在
cat("检查函数是否存在:\n")
cat("estimate_model_memory:", exists("estimate_model_memory"), "\n")
cat("check_memory_available:", exists("check_memory_available"), "\n")

# 测试通过 .Call 直接调用
cat("\n直接通过 .Call 调用:\n")
tryCatch({
  result <- .Call("c_r_estimate_model_memory", "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf")
  cat("✓ c_r_estimate_model_memory 可用，结果:", result, "\n")
}, error = function(e) {
  cat("✗ c_r_estimate_model_memory 失败:", e$message, "\n")
})

tryCatch({
  result <- .Call("c_r_check_memory_available", 1e9)  # 测试1GB
  cat("✓ c_r_check_memory_available 可用，结果:", result, "\n")
}, error = function(e) {
  cat("✗ c_r_check_memory_available 失败:", e$message, "\n")
})

# 检查所有可用的C函数
cat("\n检查所有已注册的C函数:\n")
all_symbols <- getDLLRegisteredRoutines("newrllama4")
if (length(all_symbols) > 0) {
  for (i in seq_along(all_symbols)) {
    cat(names(all_symbols)[i], ":\n")
    print(all_symbols[[i]])
  }
} else {
  cat("没有找到已注册的C函数\n")
}

# 检查 .Call 接口
cat("\n检查 .Call 接口:\n")
call_symbols <- getNativeSymbolInfo("c_r_estimate_model_memory", PACKAGE = "newrllama4")
cat("c_r_estimate_model_memory 符号信息:\n")
print(call_symbols)