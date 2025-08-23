#!/usr/bin/env Rscript

# 检查模型的元数据信息
library(newrllama4)

cat("=== 检查模型元数据 ===\n\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf"

if (!file.exists(model_path)) {
  cat("❌ 模型文件不存在\n")
  quit(status = 1)
}

cat("📁 文件信息:\n")
cat(sprintf("  路径: %s\n", model_path))
cat(sprintf("  大小: %.1f MB\n", file.info(model_path)$size / (1024*1024)))

if (!lib_is_installed()) {
  install_newrllama()
}

tryCatch({
  cat("\n📥 加载模型以获取详细信息...\n")
  model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 2L)  # 使用更多输出
  
  cat("\n🔍 模型加载成功！从加载信息中我们能看到:\n")
  cat("- 模型架构和参数\n")
  cat("- 词汇表信息\n") 
  cat("- 特殊token信息\n")
  
  # 测试tokenizer看看特殊token
  cat("\n🔤 测试特殊token:\n")
  
  test_tokens <- c("<s>", "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", 
                   "<|im_start|>", "<|im_end|>", "<start_of_turn>", "<end_of_turn>")
  
  for (token in test_tokens) {
    tryCatch({
      tokenized <- tokenize(model, token)
      cat(sprintf("  %s: tokenID=%s\n", token, paste(tokenized, collapse=",")))
    }, error = function(e) {
      cat(sprintf("  %s: 无法tokenize\n", token))
    })
  }
  
  cat("\n📝 测试内置模板信息:\n")
  
  # 简单对话测试
  simple_msg <- list(list(role = "user", content = "Hi"))
  
  cat("使用简单消息测试各函数:\n")
  
  # apply_chat_template (自动)
  result_auto <- apply_chat_template(model, simple_msg)
  cat(sprintf("apply_chat_template (auto): '%s'\n", gsub("\n", "\\n", result_auto)))
  
  # smart_chat_template  
  result_smart <- smart_chat_template(model, simple_msg)
  cat(sprintf("smart_chat_template: '%s'\n", gsub("\n", "\\n", result_smart)))
  
  rm(model)
  backend_free()
  
}, error = function(e) {
  cat("❌ 测试失败:", e$message, "\n")
  tryCatch(backend_free(), error = function(e2) {})
})

cat("\n💡 分析结论:\n")
cat("基于观察到的ChatML格式 (<|im_start|>, <|im_end|>)，\n")
cat("这个文件可能不是真正的Llama 2模型，而是:\n")
cat("1. 被转换过的模型（使用了ChatML格式）\n")
cat("2. 错误标记的模型文件\n")
cat("3. 经过特殊fine-tuning的版本\n")
cat("\n📋 检查完成!\n")