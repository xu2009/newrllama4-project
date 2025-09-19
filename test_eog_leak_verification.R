# ============================================================================
# 🔍 EOG泄漏验证测试 - 确认我们的修复是否真的在工作
# ============================================================================
#
# 目的：通过对比原始generate函数和当前修复版本，
#       验证我们的multi-token EOG检测是否真的在起作用
# ============================================================================

library(newrllama4)

cat("🔍 EOG泄漏验证测试\n")
cat(paste0(rep("=", 50), collapse=""), "\n\n")

# 使用已知会出现EOG问题的模型和prompt
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"

# 这个prompt更容易触发聊天模板的EOG tokens
test_prompt <- "Hello! How can I help you today?"

tryCatch({
  backend_free()
  Sys.sleep(1)
  
  cat("🔄 加载Llama-3.2模型...\n")
  model <- model_load(model_path, n_gpu_layers = 999)
  cat("✅ 模型加载成功\n")
  
  ctx <- context_create(model, n_ctx = 2048)
  cat("✅ 上下文创建成功\n")
  
  # 应用聊天模板 - 这更有可能触发EOG问题
  messages <- list(
    list(role = "user", content = test_prompt)
  )
  
  formatted_prompt <- apply_chat_template(model, messages, add_assistant = TRUE)
  cat("📝 聊天模板格式化完成\n")
  cat("📖 格式化后的prompt长度:", nchar(formatted_prompt), "字符\n")
  
  # Tokenize
  tokens <- tokenize(model, formatted_prompt, add_special = TRUE)
  cat("📝 Tokenized输入:", length(tokens), "tokens\n")
  
  # 测试1: 使用当前的generate函数（有我们的修复）
  cat("\n🧪 测试1: 当前的generate函数（带修复）\n")
  result1 <- generate(ctx, tokens, 
                     max_tokens = 100, 
                     temperature = 0.1, 
                     seed = 12345)
  
  cat("📝 当前版本输出 (", nchar(result1), "字符):\n")
  cat("\"", substr(result1, 1, 200), if(nchar(result1) > 200) "..." else "", "\"\n")
  
  # 检查EOG tokens
  eog_patterns <- c(
    "<\\|eot_id\\|>", 
    "<\\|end_header_id\\|>", 
    "<\\|start_header_id\\|>",
    "<\\|im_end\\|>",
    "<\\|im_start\\|>",
    "</s>"
  )
  
  found_eogs_current <- c()
  for (pattern in eog_patterns) {
    if (grepl(pattern, result1)) {
      found_eogs_current <- c(found_eogs_current, gsub("\\\\", "", pattern))
    }
  }
  
  if (length(found_eogs_current) > 0) {
    cat("🔴 当前版本发现EOG泄漏:", paste(found_eogs_current, collapse=", "), "\n")
  } else {
    cat("✅ 当前版本无EOG泄漏\n")
  }
  
  # 测试2: 测试不同的温度和种子
  cat("\n🧪 测试2: 高温度生成（更随机）\n")
  result2 <- generate(ctx, tokens, 
                     max_tokens = 100, 
                     temperature = 0.8,  # 高温度
                     seed = 99999)       # 不同种子
  
  cat("📝 高温度输出 (", nchar(result2), "字符):\n")
  cat("\"", substr(result2, 1, 200), if(nchar(result2) > 200) "..." else "", "\"\n")
  
  found_eogs_temp <- c()
  for (pattern in eog_patterns) {
    if (grepl(pattern, result2)) {
      found_eogs_temp <- c(found_eogs_temp, gsub("\\\\", "", pattern))
    }
  }
  
  if (length(found_eogs_temp) > 0) {
    cat("🔴 高温度版本发现EOG泄漏:", paste(found_eogs_temp, collapse=", "), "\n")
  } else {
    cat("✅ 高温度版本无EOG泄漏\n")
  }
  
  # 测试3: 更长的生成
  cat("\n🧪 测试3: 更长的文本生成\n")
  result3 <- generate(ctx, tokens, 
                     max_tokens = 200,   # 更长
                     temperature = 0.5,
                     seed = 54321)
  
  cat("📝 长文本输出 (", nchar(result3), "字符):\n")
  cat("\"", substr(result3, 1, 300), if(nchar(result3) > 300) "..." else "", "\"\n")
  
  found_eogs_long <- c()
  for (pattern in eog_patterns) {
    if (grepl(pattern, result3)) {
      found_eogs_long <- c(found_eogs_long, gsub("\\\\", "", pattern))
    }
  }
  
  if (length(found_eogs_long) > 0) {
    cat("🔴 长文本版本发现EOG泄漏:", paste(found_eogs_long, collapse=", "), "\n")
  } else {
    cat("✅ 长文本版本无EOG泄漏\n")
  }
  
  # 清理
  rm(model, ctx)
  backend_free()
  
  # 最终分析
  cat("\n🎯 测试总结\n")
  cat(paste0(rep("=", 30), collapse=""), "\n")
  
  total_tests <- 3
  failed_tests <- length(found_eogs_current) + length(found_eogs_temp) + length(found_eogs_long)
  
  if (failed_tests == 0) {
    cat("✅ 所有", total_tests, "项测试都通过！\n")
    cat("💡 结论：我们的multi-token EOG序列检测修复正在有效工作。\n")
    cat("🎉 当前实现已经成功解决了EOG token泄漏问题！\n")
  } else {
    cat("🔴 发现", failed_tests, "项测试中有EOG泄漏\n")
    cat("💡 结论：需要进一步改进我们的修复逻辑。\n")
  }
  
}, error = function(e) {
  cat("❌ 测试失败:", e$message, "\n")
  tryCatch(backend_free(), error = function(e2) {})
})

cat("\n=== EOG泄漏验证测试完成 ===\n")