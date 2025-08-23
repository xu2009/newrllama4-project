#!/usr/bin/env Rscript

# 详细解释三个模板函数的区别
library(newrllama4)

cat("=== 三个聊天模板函数的区别 ===\n\n")

# 使用Llama 3.2模型来测试，因为它工作正常
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"

if (!file.exists(model_path)) {
  cat("❌ 模型文件不存在\n")
  quit(status = 1)
}

# 测试消息
messages <- list(
  list(role = "system", content = "You are helpful."),
  list(role = "user", content = "Hello!"),
  list(role = "assistant", content = "Hi there!"),
  list(role = "user", content = "How are you?")
)

if (!lib_is_installed()) {
  install_newrllama()
}

tryCatch({
  cat("📥 加载 Llama 3.2 模型...\n")
  model <- model_load(model_path, n_gpu_layers = 0L, verbosity = 0L)
  
  cat("\n\n")
  cat("📋 测试三个函数的详细行为:\n")
  cat(strrep("=", 70), "\n")
  
  # 1. apply_chat_template - 主要函数
  cat("1️⃣ apply_chat_template() - 主要聊天模板函数\n")
  cat(strrep("-", 50), "\n")
  
  result1 <- apply_chat_template(model, messages)
  cat("✨ 功能说明:\n")
  cat("  - 使用模型内置的聊天模板\n")
  cat("  - 支持自定义模板参数\n")
  cat("  - 默认添加 assistant 提示符用于生成\n")
  cat("  - 这是最通用和推荐的函数\n")
  cat(sprintf("📏 输出长度: %d 字符\n", nchar(result1)))
  cat("🔍 输出预览 (前100字符):\n")
  cat("  ", substr(gsub("\n", "\\n", result1), 1, 100), "...\n")
  
  # 测试参数选项
  cat("\n🔧 测试参数选项:\n")
  
  # 不添加assistant提示符
  result1_no_assistant <- apply_chat_template(model, messages, add_assistant = FALSE)
  cat(sprintf("  - add_assistant=FALSE: %d 字符 (少了 %d 字符)\n", 
              nchar(result1_no_assistant), 
              nchar(result1) - nchar(result1_no_assistant)))
  
  # 2. smart_chat_template - 智能选择函数  
  cat("\n2️⃣ smart_chat_template() - 智能模板选择函数\n")
  cat(strrep("-", 50), "\n")
  
  result2 <- smart_chat_template(model, messages)
  cat("✨ 功能说明:\n")
  cat("  - 智能选择最适合的模板格式\n")
  cat("  - 可能包含优化逻辑\n")  
  cat("  - 可能对不同类型的消息有特殊处理\n")
  cat("  - 内部实现可能与apply_chat_template不同\n")
  cat(sprintf("📏 输出长度: %d 字符\n", nchar(result2)))
  cat("🔍 输出预览 (前100字符):\n")
  cat("  ", substr(gsub("\n", "\\n", result2), 1, 100), "...\n")
  
  # 比较差异
  if (identical(result1, result2)) {
    cat("🔄 与apply_chat_template结果: 完全相同\n")
  } else {
    cat(sprintf("🔄 与apply_chat_template差异: %d 字符\n", nchar(result2) - nchar(result1)))
    
    # 查找具体差异
    if (nchar(result1) != nchar(result2)) {
      cat("  - 长度不同，可能有格式优化\n")
    }
  }
  
  # 3. apply_gemma_chat_template - Gemma专用函数
  cat("\n3️⃣ apply_gemma_chat_template() - Gemma专用模板函数\n")
  cat(strrep("-", 50), "\n")
  
  tryCatch({
    result3 <- apply_gemma_chat_template(model, messages)
    cat("✨ 功能说明:\n")
    cat("  - 专门为Google Gemma模型设计\n") 
    cat("  - 使用Gemma特有的聊天格式\n")
    cat("  - 对非Gemma模型可能不适用\n")
    cat(sprintf("📏 输出长度: %d 字符\n", nchar(result3)))
    cat("🔍 输出预览 (前100字符):\n")
    cat("  ", substr(gsub("\n", "\\n", result3), 1, 100), "...\n")
  }, error = function(e) {
    cat("❌ 函数失败 (预期行为):\n")
    cat(sprintf("  错误: %s\n", e$message))
    cat("💡 说明: 这个函数专门用于Gemma模型，\n")
    cat("        在非Gemma模型上使用会失败。\n")
  })
  
  rm(model)
  backend_free()
  
}, error = function(e) {
  cat("❌ 测试失败:", e$message, "\n")
  tryCatch(backend_free(), error = function(e2) {})
})

cat("\n\n")
cat("📊 函数对比总结:\n")
cat(strrep("=", 70), "\n")

cat("🔸 apply_chat_template():\n")
cat("  ├─ 用途: 通用聊天模板应用函数\n")
cat("  ├─ 适用: 所有支持聊天模板的模型\n")
cat("  ├─ 特点: 功能完整，支持参数自定义\n")
cat("  └─ 推荐: ⭐⭐⭐⭐⭐ (首选)\n")

cat("\n🔸 smart_chat_template():\n")
cat("  ├─ 用途: 智能模板选择和优化\n")
cat("  ├─ 适用: 大多数模型\n") 
cat("  ├─ 特点: 可能包含优化算法\n")
cat("  └─ 推荐: ⭐⭐⭐⭐ (备选)\n")

cat("\n🔸 apply_gemma_chat_template():\n")
cat("  ├─ 用途: Gemma模型专用模板\n")
cat("  ├─ 适用: 仅Google Gemma系列模型\n")
cat("  ├─ 特点: 特化功能，格式固定\n")
cat("  └─ 推荐: ⭐⭐ (特定场景)\n")

cat("\n💡 使用建议:\n")
cat("1. 一般情况下使用 apply_chat_template()\n")
cat("2. 需要优化时可尝试 smart_chat_template()\n")
cat("3. 使用Gemma模型时使用 apply_gemma_chat_template()\n")

cat("\n📋 函数对比完成!\n")