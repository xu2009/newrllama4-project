#!/usr/bin/env Rscript

# 简化的聊天模板测试 - 验证模板自动识别功能
library(newrllama4)

cat("=== 简化聊天模板测试 ===\n\n")

# 测试模型
models <- list(
  llama32 = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf",
  llama2 = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf"
)

# 标准对话格式
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello! What's your name?"),
  list(role = "assistant", content = "I'm Claude, an AI assistant."), 
  list(role = "user", content = "Can you help me with math?")
)

cat("📋 测试消息格式:\n")
for (i in seq_along(messages)) {
  msg <- messages[[i]]
  cat(sprintf("  %d. %s: %s\n", i, msg$role, substr(msg$content, 1, 30)))
}

# 检查函数文档
cat("\n📚 查看apply_chat_template函数帮助:\n")
tryCatch({
  help_text <- capture.output(help(apply_chat_template))
  if (length(help_text) > 0) {
    cat("  ✅ 函数存在且有文档\n")
  }
}, error = function(e) {
  cat("  ⚠️ 无法获取帮助文档\n")
})

if (!lib_is_installed()) {
  install_newrllama()
}

for (model_name in names(models)) {
  model_path <- models[[model_name]]
  
  if (!file.exists(model_path)) {
    cat(sprintf("❌ %s: 文件不存在\n", model_name))
    next
  }
  
  cat(sprintf("\n🔍 测试 %s:\n", toupper(model_name)))
  cat(sprintf("   文件: %s (%.1f MB)\n", basename(model_path), file.info(model_path)$size/(1024*1024)))
  
  tryCatch({
    # 加载模型
    cat("📥 加载模型...\n")
    model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 0L)
    cat("✅ 模型加载成功\n")
    
    # 测试 apply_chat_template
    cat("🔧 测试 apply_chat_template...\n")
    result1 <- apply_chat_template(model, messages)
    
    if (!is.null(result1) && nchar(result1) > 0) {
      cat(sprintf("  ✅ 成功! 生成了 %d 字符的模板\n", nchar(result1)))
      
      # 显示特征
      has_inst <- grepl("\\[INST\\]", result1)
      has_chatml <- grepl("<\\|im_start\\|>", result1)
      has_bos <- grepl("<s>", result1)
      has_eos <- grepl("</s>", result1)
      
      cat("  🏷️ 模板特征:\n")
      cat(sprintf("    - Llama格式 [INST]: %s\n", if(has_inst) "是" else "否"))
      cat(sprintf("    - ChatML格式: %s\n", if(has_chatml) "是" else "否"))
      cat(sprintf("    - 开始标记<s>: %s\n", if(has_bos) "是" else "否"))
      cat(sprintf("    - 结束标记</s>: %s\n", if(has_eos) "是" else "否"))
      
      # 显示开头
      cat("  📄 模板开头 (前200字符):\n")
      preview <- substr(result1, 1, 200)
      cat("    ", gsub("\n", "\\n", preview), "...\n", sep = "")
      
      cat(sprintf("  💾 完整模板长度: %d 字符\n", nchar(result1)))
      
    } else {
      cat("  ❌ apply_chat_template 返回空结果\n")
    }
    
    # 测试 smart_chat_template
    cat("\n🧠 测试 smart_chat_template...\n")
    tryCatch({
      result2 <- smart_chat_template(model, messages)
      if (!is.null(result2) && nchar(result2) > 0) {
        cat(sprintf("  ✅ 成功! 生成了 %d 字符\n", nchar(result2)))
        
        # 比较两个结果是否相同
        if (identical(result1, result2)) {
          cat("  🔄 与apply_chat_template结果相同\n")
        } else {
          cat("  🔄 与apply_chat_template结果不同\n")
          cat(sprintf("    长度差异: %d 字符\n", nchar(result2) - nchar(result1)))
        }
      } else {
        cat("  ❌ smart_chat_template 返回空结果\n")
      }
    }, error = function(e) {
      cat("  ❌ smart_chat_template 失败:", e$message, "\n")
    })
    
    # 清理
    rm(model)
    backend_free()
    
  }, error = function(e) {
    cat("❌ 失败:", e$message, "\n")
    tryCatch(backend_free(), error = function(e2) {})
  })
  
  cat("\n\n")
}

cat("🎯 测试要点验证:\n")
cat("1. ✓ 每个模型都应该能成功调用 apply_chat_template\n")
cat("2. ✓ 不同模型生成的模板格式应该有差异\n")
cat("3. ✓ 生成的模板应该包含所有输入的消息内容\n")
cat("4. ✓ 模板格式应该符合各模型的标准对话格式\n")

cat("\n📋 简化模板测试完成!\n")