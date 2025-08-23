#!/usr/bin/env Rscript

# 测试apply_template自动识别不同模型内置模板功能 (健壮版本)
library(newrllama4)

cat("=== 测试apply_template自动模板识别功能 ===\n\n")

# 定义测试模型路径
models <- list(
  llama2 = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf", 
  llama32 = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf",
  deepseek = "/Users/yaoshengleo/Desktop/gguf模型/DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf"
)

# 检查模型文件是否存在
cat("📋 检查模型文件存在性:\n")
valid_models <- list()
for (name in names(models)) {
  path <- models[[name]]
  exists <- file.exists(path)
  cat(sprintf("  %s %s: %s\n", 
              if(exists) "✅" else "❌", 
              name, 
              if(exists) "存在" else "不存在"))
  if (exists) {
    valid_models[[name]] <- path
  }
}

if (length(valid_models) == 0) {
  cat("❌ 没有找到有效的模型文件，退出测试\n")
  quit(status = 1)
}

# 确保后端库已安装
if (!lib_is_installed()) {
  cat("\n正在安装newrllama后端库...\n")
  install_newrllama()
}

cat(sprintf("\n🧪 开始测试 %d 个有效模型的模板自动识别...\n\n", length(valid_models)))

# 测试消息 - 标准对话格式
test_messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello! Can you introduce yourself?"),
  list(role = "assistant", content = "Hi! I'm an AI assistant created to be helpful, harmless, and honest."),
  list(role = "user", content = "What's the weather like today?")
)

# 测试每个模型
test_results <- list()

for (model_name in names(valid_models)) {
  model_path <- valid_models[[model_name]]
  
  cat(sprintf("🔍 测试 %s 模型:\n", toupper(model_name)))
  cat(sprintf("   路径: %s\n", basename(model_path)))
  cat(sprintf("   大小: %.1f MB\n", file.info(model_path)$size / (1024*1024)))
  cat(strrep("-", 60), "\n")
  
  # 尝试加载和测试模型
  model_success <- FALSE
  tryCatch({
    # 加载模型 (使用verbosity=0来减少输出)
    cat("📥 正在加载模型...\n")
    model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 0L)
    model_success <- TRUE
    cat("✅ 模型加载成功!\n")
    
    # 测试apply_template自动识别
    cat("🔧 测试apply_template自动模板识别...\n")
    
    # 不指定template参数，让函数自动识别
    result <- apply_template(model, test_messages)
    
    # 记录结果
    test_results[[model_name]] <- list(
      model_file = basename(model_path),
      model_loaded = TRUE,
      template_applied = !is.null(result),
      formatted_prompt = result,
      prompt_length = nchar(result)
    )
    
    # 显示结果
    cat("✅ 模板应用成功!\n")
    
    # 分析模板特征
    features <- analyze_template_features(result)
    cat("📋 检测到的模板特征:", paste(features, collapse = ", "), "\n")
    cat("📏 生成提示词长度:", nchar(result), "字符\n")
    
    # 显示模板格式的关键部分
    cat("🔍 模板格式预览:\n")
    show_template_preview(result)
    
    cat("✅ 测试完成!\n")
    
  }, error = function(e) {
    cat("❌ 测试失败:", e$message, "\n")
    test_results[[model_name]] <- list(
      model_file = basename(model_path),
      model_loaded = model_success,
      template_applied = FALSE,
      error = e$message
    )
  })
  
  # 清理资源
  tryCatch({
    if (model_success) {
      rm(model)
    }
    backend_free()
  }, error = function(e) {
    cat("⚠️ 资源清理警告:", e$message, "\n")
  })
  
  cat("\n")
  Sys.sleep(2)  # 给系统一点时间恢复
}

# 辅助函数：分析模板特征
analyze_template_features <- function(prompt) {
  features <- c()
  
  # 检测常见模板格式标记
  if (grepl("<\\|im_start\\|>", prompt)) features <- c(features, "ChatML格式")
  if (grepl("\\[INST\\]", prompt)) features <- c(features, "Llama指令格式")
  if (grepl("\\[/INST\\]", prompt)) features <- c(features, "Llama响应格式")
  if (grepl("<s>", prompt)) features <- c(features, "开始标记<s>")
  if (grepl("</s>", prompt)) features <- c(features, "结束标记</s>")
  if (grepl("System:", prompt)) features <- c(features, "System前缀")
  if (grepl("Human:", prompt)) features <- c(features, "Human前缀")
  if (grepl("Assistant:", prompt)) features <- c(features, "Assistant前缀")
  if (grepl("###", prompt)) features <- c(features, "###分隔符")
  if (grepl("USER:", prompt)) features <- c(features, "USER前缀")
  if (grepl("ASSISTANT:", prompt)) features <- c(features, "ASSISTANT前缀")
  
  if (length(features) == 0) features <- c("标准文本格式")
  
  return(features)
}

# 辅助函数：显示模板预览
show_template_preview <- function(prompt) {
  # 分割成行并显示前几行和后几行
  lines <- strsplit(prompt, "\n")[[1]]
  total_lines <- length(lines)
  
  cat("   开头部分:\n")
  for (i in 1:min(3, total_lines)) {
    line_preview <- substr(lines[i], 1, 80)
    cat(sprintf("   %d: %s%s\n", i, line_preview, 
                if(nchar(lines[i]) > 80) "..." else ""))
  }
  
  if (total_lines > 6) {
    cat("   ...\n")
    cat("   结尾部分:\n")
    for (i in max(total_lines-2, 4):total_lines) {
      line_preview <- substr(lines[i], 1, 80)
      cat(sprintf("   %d: %s%s\n", i, line_preview,
                  if(nchar(lines[i]) > 80) "..." else ""))
    }
  } else if (total_lines > 3) {
    for (i in 4:total_lines) {
      line_preview <- substr(lines[i], 1, 80)
      cat(sprintf("   %d: %s%s\n", i, line_preview,
                  if(nchar(lines[i]) > 80) "..." else ""))
    }
  }
}

# 总结测试结果
cat(strrep("=", 60), "\n")
cat("📊 测试结果总结:\n")
cat(strrep("=", 60), "\n")

successful_tests <- 0
total_tests <- length(test_results)

for (model_name in names(test_results)) {
  result <- test_results[[model_name]]
  cat(sprintf("\n🔸 %s (%s):\n", toupper(model_name), result$model_file))
  
  if (!is.null(result$model_loaded) && result$model_loaded) {
    cat("  ✅ 模型加载: 成功\n")
    
    if (!is.null(result$template_applied) && result$template_applied) {
      cat("  ✅ 模板自动识别: 成功\n")
      cat("  📏 提示词长度:", result$prompt_length, "字符\n")
      successful_tests <- successful_tests + 1
      
      # 快速特征分析
      features <- analyze_template_features(result$formatted_prompt)
      cat("  🏷️ 主要特征:", paste(features[1:min(2, length(features))], collapse = ", "), "\n")
      
    } else {
      cat("  ❌ 模板自动识别: 失败\n")
    }
  } else {
    cat("  ❌ 模型加载: 失败\n")
    if (!is.null(result$error)) {
      # 简化错误信息显示
      error_short <- substr(result$error, 1, 100)
      cat("  🚨 错误:", error_short, "...\n")
    }
  }
}

cat(sprintf("\n📈 总体结果: %d/%d 个模型测试成功 (%.1f%%)\n", 
            successful_tests, total_tests, 
            (successful_tests/total_tests) * 100))

cat("\n🎯 验证要点:\n")
cat("✓ 每个模型应该能自动识别其内置聊天模板\n")
cat("✓ 不同模型生成的模板格式应该有明显差异\n")
cat("✓ apply_template函数应该无需手动指定template参数\n")
cat("✓ 生成的提示词应该包含正确的对话格式标记\n")

cat("\n📋 模板自动识别测试完成!\n")