#!/usr/bin/env Rscript

# 测试聊天模板自动识别功能 - 使用正确的函数名
library(newrllama4)

cat("=== 测试聊天模板自动识别功能 ===\n\n")

# 定义测试模型路径 (先测试较小的模型)
models <- list(
  llama32 = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf",
  llama2 = "/Users/yaoshengleo/Desktop/gguf模型/llama-2-7b-chat.Q8_0.gguf"
  # DeepSeek模型暂时跳过，因为文件可能损坏
)

# 检查模型文件
cat("📋 检查模型文件:\n")
valid_models <- list()
for (name in names(models)) {
  path <- models[[name]]
  if (file.exists(path)) {
    valid_models[[name]] <- path
    cat(sprintf("  ✅ %s: %.1f MB\n", name, file.info(path)$size / (1024*1024)))
  } else {
    cat(sprintf("  ❌ %s: 文件不存在\n", name))
  }
}

if (length(valid_models) == 0) {
  cat("❌ 没有找到有效的模型文件\n")
  quit(status = 1)
}

# 确保后端库已安装
if (!lib_is_installed()) {
  cat("\n正在安装newrllama后端库...\n")
  install_newrllama()
}

cat(sprintf("\n🧪 开始测试 %d 个模型的聊天模板功能...\n", length(valid_models)))

# 标准测试对话
test_messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello! Can you introduce yourself?"),
  list(role = "assistant", content = "Hi! I'm an AI assistant."),
  list(role = "user", content = "What's 2+2?")
)

# 测试结果存储
test_results <- list()

for (model_name in names(valid_models)) {
  model_path <- valid_models[[model_name]]
  
  cat(sprintf("\n🔍 测试 %s 模型:\n", toupper(model_name)))
  cat(sprintf("   文件: %s\n", basename(model_path)))
  cat(strrep("-", 50), "\n")
  
  result_entry <- list(
    model_name = model_name,
    model_file = basename(model_path),
    tests = list()
  )
  
  tryCatch({
    # 加载模型
    cat("📥 加载模型...\n")
    model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 0L)
    cat("✅ 模型加载成功\n")
    
    # 测试不同的模板函数
    template_functions <- list(
      "apply_chat_template" = apply_chat_template,
      "smart_chat_template" = smart_chat_template,
      "apply_gemma_chat_template" = apply_gemma_chat_template
    )
    
    for (func_name in names(template_functions)) {
      cat(sprintf("\n🔧 测试 %s 函数...\n", func_name))
      
      tryCatch({
        # 调用模板函数
        result <- template_functions[[func_name]](model, test_messages)
        
        if (!is.null(result) && nchar(result) > 0) {
          # 分析模板特征
          features <- analyze_template_features(result)
          
          result_entry$tests[[func_name]] <- list(
            success = TRUE,
            result_length = nchar(result),
            features = features,
            preview = substr(result, 1, 200)
          )
          
          cat("  ✅ 成功!\n")
          cat(sprintf("  📏 生成长度: %d 字符\n", nchar(result)))
          cat(sprintf("  🏷️ 特征: %s\n", paste(features, collapse = ", ")))
          
          # 显示格式预览
          cat("  🔍 格式预览:\n")
          lines <- strsplit(result, "\n")[[1]]
          for (i in 1:min(3, length(lines))) {
            preview_line <- substr(lines[i], 1, 60)
            cat(sprintf("    %s%s\n", preview_line, if(nchar(lines[i]) > 60) "..." else ""))
          }
          
        } else {
          result_entry$tests[[func_name]] <- list(success = FALSE, error = "空结果")
          cat("  ❌ 返回空结果\n")
        }
        
      }, error = function(e) {
        result_entry$tests[[func_name]] <- list(success = FALSE, error = e$message)
        cat("  ❌ 失败:", e$message, "\n")
      })
    }
    
    # 清理模型
    rm(model)
    backend_free()
    
  }, error = function(e) {
    result_entry$load_error <- e$message
    cat("❌ 模型加载失败:", e$message, "\n")
    tryCatch(backend_free(), error = function(e2) {})
  })
  
  test_results[[model_name]] <- result_entry
  cat("\n")
  Sys.sleep(1)
}

# 辅助函数：分析模板特征
analyze_template_features <- function(prompt) {
  features <- c()
  
  if (grepl("<\\|im_start\\|>", prompt)) features <- c(features, "ChatML")
  if (grepl("\\[INST\\]", prompt)) features <- c(features, "Llama指令格式")
  if (grepl("<s>", prompt)) features <- c(features, "BOS标记")
  if (grepl("</s>", prompt)) features <- c(features, "EOS标记")
  if (grepl("System:", prompt)) features <- c(features, "System前缀")
  if (grepl("User:", prompt)) features <- c(features, "User前缀")
  if (grepl("Assistant:", prompt)) features <- c(features, "Assistant前缀")
  if (grepl("###", prompt)) features <- c(features, "###分隔符")
  
  if (length(features) == 0) features <- c("标准格式")
  return(features)
}

# 输出测试总结
cat(strrep("=", 60), "\n")
cat("📊 测试结果总结\n")
cat(strrep("=", 60), "\n")

for (model_name in names(test_results)) {
  result <- test_results[[model_name]]
  cat(sprintf("\n🎯 %s (%s):\n", toupper(model_name), result$model_file))
  
  if (!is.null(result$load_error)) {
    cat("  ❌ 模型加载失败\n")
    next
  }
  
  successful_functions <- 0
  total_functions <- length(result$tests)
  
  for (func_name in names(result$tests)) {
    test_result <- result$tests[[func_name]]
    if (test_result$success) {
      successful_functions <- successful_functions + 1
      cat(sprintf("  ✅ %s: 成功 (%d字符)\n", func_name, test_result$result_length))
    } else {
      cat(sprintf("  ❌ %s: 失败\n", func_name))
    }
  }
  
  cat(sprintf("  📈 成功率: %d/%d (%.1f%%)\n", 
              successful_functions, total_functions,
              (successful_functions/total_functions) * 100))
}

cat("\n🎯 关键发现:\n")
cat("• apply_chat_template: 通用聊天模板函数\n")
cat("• smart_chat_template: 智能模板选择函数\n") 
cat("• apply_gemma_chat_template: Gemma专用模板函数\n")
cat("• 每个模型应该至少有一个函数能成功生成聊天模板\n")

cat("\n📋 聊天模板测试完成!\n")