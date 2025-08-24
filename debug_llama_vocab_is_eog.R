#!/usr/bin/env Rscript

# 深度诊断：验证llama_vocab_is_eog()函数是否正确工作
library(newrllama4)

cat("=== 深度诊断llama_vocab_is_eog()函数 ===\n\n")

# 确保使用最新版本
if (!lib_is_installed()) {
  install_newrllama()
}

model_path <- "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

cat("📥 加载模型和上下文...\n")
model <- model_load(model_path, n_gpu_layers = 100L, verbosity = 0L)
ctx <- context_create(model, n_ctx = 512L, verbosity = 0L)

cat("✅ 模型和上下文准备就绪\n\n")

# 测试各种可疑的token
test_texts <- c(
  "<|eot_id|>",
  " <|eot_id|>",
  " <|eot_id|> ",
  "<|end_header_id|>",
  " <|end_header_id|>",
  " <|end_header_id|> ",
  "<|start_header_id|>",
  " <|start_header_id|>",
  " <|start_header_id|> "
)

cat("🔍 测试各种EOG token文本的token化:\n")
cat(strrep("-", 60), "\n")

for (text in test_texts) {
  cat(sprintf("文本: '%s'\n", text))
  
  # 分词
  token_ids <- tokenize(model, text)
  
  # 获取token文本
  token_texts <- c()
  for (id in token_ids) {
    token_text <- token_to_piece(model, id)
    token_texts <- c(token_texts, token_text)
  }
  
  cat(sprintf("  Token IDs: [%s]\n", paste(token_ids, collapse = ", ")))
  cat(sprintf("  Token文本: [%s]\n", paste(sapply(token_texts, function(x) sprintf("'%s'", x)), collapse = ", ")))
  
  # 检查每个token是否被llama.cpp识别为EOG
  eog_results <- c()
  for (i in seq_along(token_ids)) {
    # 这里我们需要通过C++层检查，但当前API可能没有导出
    # 我们通过观察生成行为来推断
    eog_results <- c(eog_results, "未知")
  }
  
  cat(sprintf("  EOG检测: [%s]\n", paste(eog_results, collapse = ", ")))
  cat("\n")
}

# 重点测试：生成包含这些token的序列
cat("🎯 重点测试：强制生成EOG token序列\n")
cat(strrep("-", 60), "\n")

# 创建一个会生成EOG token的prompt
test_prompt <- "The special token is <|eot_id"
cat(sprintf("测试prompt: '%s'\n", test_prompt))

tokens_in <- tokenize(model, test_prompt)
cat(sprintf("输入tokens: [%s]\n", paste(tokens_in, collapse = ", ")))

# 生成一些tokens，看看是否会被EOG检测捕获
result <- generate_tokens(model, ctx, tokens_in, max_tokens = 10L)
cat(sprintf("生成结果: '%s'\n", result))

# 如果结果包含完整的 <|eot_id|>，说明EOG检测没有工作
if (grepl("<\\|eot_id\\|>", result)) {
  cat("❌ 严重问题：llama_vocab_is_eog()没有正确识别<|eot_id|>完整token\n")
} else if (grepl("eot_id", result)) {
  cat("⚠️ 部分问题：生成了eot_id但被某种机制截断\n")
} else {
  cat("✅ 可能正常：没有生成eot_id相关内容\n")
}

cat("\n📊 诊断总结:\n")
cat("1. llama_vocab_is_eog()可能无法识别multi-token的EOG序列\n")
cat("2. 或者模型本身的EOG token定义与我们的理解不符\n")
cat("3. 需要检查模型内部的special_eog_ids集合\n")

# 清理
rm(model, ctx)
backend_free()

cat("\n✅ 深度诊断完成\n")