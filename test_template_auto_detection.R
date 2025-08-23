#!/usr/bin/env Rscript

# 测试apply_template自动识别不同模型内置模板功能
library(newrllama4)

cat("=== 测试apply_template自动模板识别功能 ===\n\n")

# 定义测试模型路径
models <- list(
  deepseek = "/Users/yaoshengleo/Desktop/gguf模型/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
  llama2 = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf", 
  llama32 = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"
)

# 检查模型文件是否存在
cat("📋 检查模型文件存在性:\n")
for (name in names(models)) {
  path <- models[[name]]
  exists <- file.exists(path)
  cat(sprintf("  %s %s: %s\n", 
              if(exists) "✅" else "❌", 
              name, 
              if(exists) "存在" else "不存在"))
  if (!exists) {
    cat("❌ 模型文件缺失，退出测试\n")
    quit(status = 1)
  }
}

# 确保后端库已安装
if (!lib_is_installed()) {
  cat("\n正在安装newrllama后端库...\n")
  install_newrllama()
}

cat("\n🧪 开始测试各模型的模板自动识别...\n\n")

# 测试消息
test_messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello! Can you introduce yourself?"),
  list(role = "assistant", content = "Hi! I'm an AI assistant created to be helpful, harmless, and honest."),
  list(role = "user", content = "What's the weather like today?")
)

# 测试每个模型
test_results <- list()

for (model_name in names(models)) {
  model_path <- models[[model_name]]
  
  cat(sprintf("🔍 测试 %s 模型:\n", toupper(model_name)))
  cat(sprintf("   路径: %s\n", basename(model_path)))
  cat(strrep("-", 60), "\n")
  
  tryCatch({
    # 加载模型 (使用verbosity=0来减少输出)
    cat("📥 加载模型...\n")
    model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 0L)
    
    # 测试apply_template自动识别
    cat("🔧 测试apply_template自动模板识别...\n")
    
    # 不指定template，让函数自动识别
    result <- apply_template(model, test_messages)
    
    # 记录结果
    test_results[[model_name]] <- list(
      model_file = basename(model_path),
      template_applied = !is.null(result),
      formatted_prompt = result
    )
    
    # 显示结果
    cat("✅ 模板应用成功!\n")
    cat("📝 生成的提示词格式:\n")
    cat("```\n")
    # 显示前500个字符来查看格式
    preview <- substr(result, 1, 500)
    cat(preview)
    if (nchar(result) > 500) {
      cat("\n... (截断显示，共", nchar(result), "字符)\n")
    }
    cat("\n```\n")
    
    # 清理模型
    rm(model)
    backend_free()
    
  }, error = function(e) {
    cat("❌ 测试失败:", e$message, "\n")
    test_results[[model_name]] <- list(
      model_file = basename(model_path),
      template_applied = FALSE,
      error = e$message
    )
    
    # 尝试清理
    tryCatch(backend_free(), error = function(e2) {})
  })
  
  cat("\n")
  Sys.sleep(1)  # 短暂暂停
}

# 总结测试结果
cat("="*60, "\n")
cat("📊 测试结果总结:\n")
cat("="*60, "\n")

for (model_name in names(test_results)) {
  result <- test_results[[model_name]]
  cat(sprintf("\n🔸 %s (%s):\n", toupper(model_name), result$model_file))
  
  if (result$template_applied) {
    cat("  ✅ 模板自动识别: 成功\n")
    
    # 分析模板特征
    prompt <- result$formatted_prompt
    
    # 检测常见模板特征
    features <- c()
    if (grepl("<\\|im_start\\|>", prompt)) features <- c(features, "ChatML格式")
    if (grepl("\\[INST\\]", prompt)) features <- c(features, "Llama格式") 
    if (grepl("<s>", prompt)) features <- c(features, "特殊标记<s>")
    if (grepl("</s>", prompt)) features <- c(features, "特殊标记</s>")
    if (grepl("System:", prompt)) features <- c(features, "System前缀")
    if (grepl("Human:", prompt)) features <- c(features, "Human前缀")
    if (grepl("Assistant:", prompt)) features <- c(features, "Assistant前缀")
    
    cat("  📋 检测到的模板特征:", if(length(features) > 0) paste(features, collapse = ", ") else "标准格式", "\n")
    cat("  📏 生成提示词长度:", nchar(prompt), "字符\n")
    
    # 显示模板格式的开头部分
    preview_start <- substr(prompt, 1, 100)
    cat("  🔍 模板开头预览:", gsub("\n", "\\n", preview_start), "...\n")
    
  } else {
    cat("  ❌ 模板自动识别: 失败\n")
    if (!is.null(result$error)) {
      cat("  🚨 错误信息:", result$error, "\n")
    }
  }
}

cat("\n" * 2)
cat("🎯 测试要点验证:\n")
cat("1. ✓ 每个模型都应该能自动识别其内置模板\n")
cat("2. ✓ 不同模型生成的模板格式应该不同\n") 
cat("3. ✓ 模板应用应该不需要手动指定template参数\n")
cat("4. ✓ 生成的提示词应该符合各模型的对话格式规范\n")

cat("\n📋 测试完成!\n")