# 全面的EOG Token泄漏诊断脚本
# 目标：找出token泄漏发生在哪一层，以及llama.cpp的EOG检测是否真正工作
library(newrllama4)

cat("=== 全面EOG Token泄漏诊断 ===\n")
cat("目标：分析'Tell me a joke'的token泄漏问题\n\n")

# ============================================================================
# 第1步：版本和环境检查
# ============================================================================
cat("📋 第1步：版本和环境检查\n")
cat(strrep("=", 50), "\n")

cat("- 后端库是否安装:", lib_is_installed(), "\n")
if (lib_is_installed()) {
  lib_path <- get_lib_path()
  cat("- 后端库路径:", lib_path, "\n")
  
  # 检查文件修改时间，确认是否为最新版本
  lib_info <- file.info(lib_path)
  cat("- 后端库修改时间:", as.character(lib_info$mtime), "\n")
  
  # 强制重新安装以确保使用最新版本
  cat("- 强制重新安装最新版本...\n")
  install_newrllama()
  cat("  安装完成\n")
} else {
  cat("❌ 后端库未安装，正在安装...\n")
  install_newrllama()
}

# 重置所有缓存状态
quick_llama_reset()

# ============================================================================
# 第2步：模型和上下文初始化
# ============================================================================
cat("\n📝 第2步：模型和上下文初始化\n")
cat(strrep("=", 50), "\n")

# 使用quick_llama默认模型
model_path <- "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

cat("使用模型: Llama-3.2-1B-Instruct-Q4_K_M.gguf\n")
cat("模型路径:", model_path, "\n")

# 加载模型（使用100层GPU，启用详细输出）
cat("正在加载模型到GPU (100层)...\n")
model <- model_load(model_path, n_gpu_layers = 100L, verbosity = 1L)

# 创建上下文
cat("正在创建上下文...\n")
ctx <- context_create(model, n_ctx = 512L, verbosity = 1L)

cat("✅ 模型和上下文初始化完成\n")

# ============================================================================
# 第3步：特殊Token分析
# ============================================================================
cat("\n🏷️ 第3步：特殊Token分析\n")
cat(strrep("=", 50), "\n")

# 获取模型的特殊token（注：当前版本可能没有导出这些函数）
cat("模型特殊token:\n")
cat("  - 注：当前版本没有导出token_eos等函数，直接分析EOG tokens\n")

# 分析常见EOG token的实际token ID
common_eog_texts <- c(
  "<|eot_id|>", 
  "<|end_header_id|>", 
  "<|start_header_id|>", 
  "<|assistant|>",
  "<|user|>",
  "<|system|>",
  "<|im_start|>", 
  "<|im_end|>",
  "<end_of_turn>",
  "</s>"
)

cat("\n常见EOG文本的token ID:\n")
eog_token_map <- list()
for (text in common_eog_texts) {
  token_ids <- tryCatch({
    tokenize(model, text, add_special = FALSE)
  }, error = function(e) {
    NULL
  })
  if (!is.null(token_ids) && length(token_ids) > 0) {
    cat("  - \"", text, "\" -> [", paste(token_ids, collapse=", "), "]\n")
    eog_token_map[[text]] <- token_ids
  } else {
    cat("  - \"", text, "\" -> 无法分词或不存在\n")
  }
}

# ============================================================================
# 第4步：Prompt格式化和分词测试
# ============================================================================
cat("\n📤 第4步：Prompt格式化和分词\n")
cat(strrep("=", 50), "\n")

# 构建与quick_llama相同的消息结构
test_prompt <- "Tell me a joke."
system_prompt <- "You are a helpful assistant."

messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = test_prompt)
)

# 应用chat template
formatted_prompt <- apply_chat_template(model, messages, add_assistant = TRUE)

cat("原始prompt:", test_prompt, "\n")
cat("系统prompt:", system_prompt, "\n")
cat("格式化后的完整prompt:\n")
cat(strrep("-", 40), "\n")
cat(formatted_prompt)
cat(strrep("-", 40), "\n")

# 分词
tokens <- tokenize(model, formatted_prompt, add_special = TRUE)
cat("Token数量:", length(tokens), "\n")
cat("前10个token:", paste(head(tokens, 10), collapse=", "), "\n")

# ============================================================================
# 第5步：底层generate()函数测试
# ============================================================================
cat("\n⚡ 第5步：底层generate()函数测试\n")
cat(strrep("=", 50), "\n")

cat("调用底层generate()函数（绕过所有清理逻辑）...\n")

# 使用相同的参数调用底层generate
raw_result <- generate(ctx, tokens, 
                      max_tokens = 100L,
                      top_k = 20L,
                      top_p = 0.9,
                      temperature = 0.7,
                      seed = 1234L)

cat("底层generate()原始输出:\n")
cat(strrep("-", 40), "\n")
cat("\"", raw_result, "\"\n")
cat(strrep("-", 40), "\n")
cat("输出长度:", nchar(raw_result), "字符\n")

# 详细分析输出中的EOG token
cat("\n🔍 底层输出EOG Token分析:\n")
eog_found <- FALSE
for (text in names(eog_token_map)) {
  # 使用固定字符串匹配而不是正则表达式，更精确
  if (grepl(text, raw_result, fixed = TRUE)) {
    positions <- gregexpr(text, raw_result, fixed = TRUE)[[1]]
    cat("  ❌ 发现 \"", text, "\" 在位置:", paste(positions, collapse=", "), "\n")
    eog_found <- TRUE
  }
}
if (!eog_found) {
  cat("  ✅ 底层输出未发现已知的EOG tokens\n")
}

# 如果底层输出就包含EOG token，说明C++修复没有生效
if (eog_found) {
  cat("\n🚨 严重问题：底层generate()就在输出EOG tokens！\n")
  cat("   这表明C++层的EOG token检测没有正确工作。\n")
}

# ============================================================================
# 第6步：.clean_output()函数测试
# ============================================================================
cat("\n🧹 第6步：.clean_output()函数测试\n")
cat(strrep("=", 50), "\n")

# 访问内部的.clean_output函数
clean_output_func <- newrllama4:::.clean_output
cleaned_result <- clean_output_func(raw_result)

cat(".clean_output()处理后的输出:\n")
cat(strrep("-", 40), "\n")
cat("\"", cleaned_result, "\"\n")
cat(strrep("-", 40), "\n")
cat("清理后长度:", nchar(cleaned_result), "字符\n")

# 检查清理效果
cat("\n🔍 清理后EOG Token分析:\n")
cleaned_eog_found <- FALSE
for (text in names(eog_token_map)) {
  if (grepl(text, cleaned_result, fixed = TRUE)) {
    positions <- gregexpr(text, cleaned_result, fixed = TRUE)[[1]]
    cat("  ❌ 清理后仍存在 \"", text, "\" 在位置:", paste(positions, collapse=", "), "\n")
    cleaned_eog_found <- TRUE
  }
}
if (!cleaned_eog_found) {
  cat("  ✅ 清理后未发现已知的EOG tokens\n")
}

# 分析清理函数是否有遗漏
if (eog_found && cleaned_eog_found) {
  cat("\n🚨 .clean_output()函数需要增强！\n")
  cat("   某些EOG tokens没有被清理规则覆盖。\n")
}

# ============================================================================
# 第7步：完整quick_llama()测试
# ============================================================================
cat("\n🚀 第7步：完整quick_llama()测试\n")
cat(strrep("=", 50), "\n")

# 清理之前的资源
rm(model, ctx)
backend_free()
quick_llama_reset()

cat("调用完整的quick_llama()函数...\n")

final_result <- quick_llama("Tell me a joke.", 
                           n_gpu_layers = 100L,
                           max_tokens = 100L,
                           verbosity = 1L,
                           seed = 1234L,
                           auto_format = TRUE)

cat("quick_llama()最终输出:\n")
cat(strrep("-", 40), "\n")
cat("\"", final_result, "\"\n")
cat(strrep("-", 40), "\n")
cat("最终输出长度:", nchar(final_result), "字符\n")

# 最终检查
cat("\n🔍 最终输出EOG Token分析:\n")
final_eog_found <- FALSE
for (text in names(eog_token_map)) {
  if (grepl(text, final_result, fixed = TRUE)) {
    positions <- gregexpr(text, final_result, fixed = TRUE)[[1]]
    cat("  ❌ 最终仍存在 \"", text, "\" 在位置:", paste(positions, collapse=", "), "\n")
    final_eog_found <- TRUE
  }
}
if (!final_eog_found) {
  cat("  ✅ 最终输出已清理所有已知EOG tokens\n")
}

# ============================================================================
# 第8步：问题定位总结
# ============================================================================
cat("\n📊 第8步：问题定位总结\n")
cat(strrep("=", 50), "\n")

cat("诊断结果:\n")
if (eog_found) {
  cat("  ❌ 问题层级：C++层（底层generate函数）\n")
  cat("  🔧 需要修复：newrllama_generate()中的EOG token检测逻辑\n")
  if (cleaned_eog_found) {
    cat("  ⚠️  附加问题：.clean_output()函数也需要增强\n")
  }
} else if (cleaned_eog_found) {
  cat("  ❌ 问题层级：R层（.clean_output函数）\n")  
  cat("  🔧 需要修复：增强.clean_output()的清理规则\n")
} else if (final_eog_found) {
  cat("  ❌ 问题层级：quick_llama逻辑\n")
  cat("  🔧 需要修复：quick_llama函数的处理流程\n") 
} else {
  cat("  ✅ 问题已解决！所有层级都正确处理了EOG tokens\n")
}

cat("\n下一步建议:\n")
if (eog_found) {
  cat("  1. 验证v1.0.73的C++代码是否正确编译和加载\n")
  cat("  2. 检查llama_vocab_is_eog()函数是否正确工作\n")
  cat("  3. 可能需要修复C++层的token检测逻辑\n")
} else if (cleaned_eog_found || final_eog_found) {
  cat("  1. 增强.clean_output()函数的清理规则\n")
  cat("  2. 确保覆盖所有可能的EOG token模式\n")
}

cat("\n✅ 全面诊断完成！\n")
cat("💡 基于上述分析结果制定修复方案。\n")