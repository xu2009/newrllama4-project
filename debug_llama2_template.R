#!/usr/bin/env Rscript

# 调试Llama 2模板格式问题
library(newrllama4)

cat("=== 调试Llama 2模板格式 ===\n\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf"

if (!file.exists(model_path)) {
  cat("❌ 模型文件不存在\n")
  quit(status = 1)
}

# 简单的测试消息
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello!")
)

cat("📋 测试消息:\n")
for (i in seq_along(messages)) {
  msg <- messages[[i]]
  cat(sprintf("  %s: %s\n", msg$role, msg$content))
}

if (!lib_is_installed()) {
  install_newrllama()
}

tryCatch({
  cat("\n📥 加载Llama 2模型...\n")
  model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 1L)
  
  cat("\n🔍 测试各种模板函数的输出:\n")
  cat(strrep("=", 60), "\n")
  
  # 1. apply_chat_template
  cat("1️⃣ apply_chat_template结果:\n")
  cat(strrep("-", 40), "\n")
  result1 <- apply_chat_template(model, messages)
  cat("长度:", nchar(result1), "字符\n")
  cat("原始输出:\n")
  cat(result1)
  cat("\n可视化 (显示换行符):\n")
  cat(gsub("\n", "\\n", result1), "\n")
  
  # 2. smart_chat_template  
  cat("\n2️⃣ smart_chat_template结果:\n")
  cat(strrep("-", 40), "\n")
  result2 <- smart_chat_template(model, messages)
  cat("长度:", nchar(result2), "字符\n")
  cat("原始输出:\n")
  cat(result2)
  cat("\n可视化 (显示换行符):\n")
  cat(gsub("\n", "\\n", result2), "\n")
  
  # 3. apply_gemma_chat_template
  cat("\n3️⃣ apply_gemma_chat_template结果:\n")
  cat(strrep("-", 40), "\n")
  tryCatch({
    result3 <- apply_gemma_chat_template(model, messages)
    cat("长度:", nchar(result3), "字符\n")
    cat("原始输出:\n")
    cat(result3)
    cat("\n可视化 (显示换行符):\n")
    cat(gsub("\n", "\\n", result3), "\n")
  }, error = function(e) {
    cat("❌ 失败:", e$message, "\n")
  })
  
  # 4. 手动指定Llama 2标准模板
  cat("\n4️⃣ 手动指定Llama 2标准模板:\n")
  cat(strrep("-", 40), "\n")
  
  # Llama 2官方模板
  llama2_template <- "<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]"
  
  tryCatch({
    result4 <- apply_chat_template(model, messages, template = llama2_template)
    cat("长度:", nchar(result4), "字符\n")
    cat("原始输出:\n")
    cat(result4)
    cat("\n可视化 (显示换行符):\n")
    cat(gsub("\n", "\\n", result4), "\n")
  }, error = function(e) {
    cat("❌ 失败:", e$message, "\n")
  })
  
  # 分析差异
  cat("\n📊 结果对比分析:\n")
  cat(strrep("=", 60), "\n")
  
  results <- list(
    "apply_chat_template" = result1,
    "smart_chat_template" = result2
  )
  
  for (name in names(results)) {
    result <- results[[name]]
    cat(sprintf("\n🔍 %s:\n", name))
    
    # 检测格式特征
    has_inst <- grepl("\\[INST\\]", result)
    has_chatml <- grepl("<\\|im_start\\|>", result)
    has_bos <- grepl("<s>", result)
    has_eos <- grepl("</s>", result)
    has_sys <- grepl("<<SYS>>", result)
    
    cat(sprintf("  - 包含[INST]: %s\n", if(has_inst) "是" else "否"))
    cat(sprintf("  - 包含<|im_start|>: %s\n", if(has_chatml) "是" else "否"))
    cat(sprintf("  - 包含<s>: %s\n", if(has_bos) "是" else "否"))
    cat(sprintf("  - 包含</s>: %s\n", if(has_eos) "是" else "否"))
    cat(sprintf("  - 包含<<SYS>>: %s\n", if(has_sys) "是" else "否"))
  }
  
  rm(model)
  backend_free()
  
}, error = function(e) {
  cat("❌ 测试失败:", e$message, "\n")
  tryCatch(backend_free(), error = function(e2) {})
})

cat("\n💡 预期的Llama 2标准格式:\n")
cat("<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>\n")
cat("\n📋 调试完成!\n")