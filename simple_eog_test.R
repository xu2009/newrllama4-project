#!/usr/bin/env Rscript

# 简化的EOG测试：专注于核心问题
library(newrllama4)

cat("=== 简化EOG问题诊断 ===\n\n")

if (!lib_is_installed()) {
  install_newrllama()
}

# 测试一个明确会产生<|eot_id|>的场景
model_path <- "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

cat("📥 加载模型...\n")
model <- model_load(model_path, n_gpu_layers = 100L, verbosity = 0L)
ctx <- context_create(model, n_ctx = 512L, verbosity = 0L)

# 测试关键问题：为什么llama_vocab_is_eog()没有工作？
cat("\n🔍 核心问题：llama_vocab_is_eog()检测\n")
cat(strrep("=", 50), "\n")

# 使用简单的生成测试
test_prompt <- "Hello"
tokens_in <- tokenize(model, test_prompt)
cat(sprintf("测试prompt: '%s'\n", test_prompt))
cat(sprintf("Token化结果: [%s]\n", paste(tokens_in, collapse = ", ")))

# 生成并观察是否包含EOG tokens  
cat("\n📤 调用底层generate (max_tokens=50):\n")
result <- generate(model, ctx, tokens_in, max_tokens = 50L)
cat(sprintf("生成结果: '%s'\n", result))
cat(sprintf("结果长度: %d 字符\n", nchar(result)))

# 关键分析：检查是否包含EOG tokens
eog_patterns <- c("<\\|eot_id\\|>", "<\\|end_header_id\\|>", "<\\|start_header_id\\|>")
found_eogs <- c()

for (pattern in eog_patterns) {
  if (grepl(pattern, result)) {
    matches <- gregexpr(pattern, result)[[1]]
    found_eogs <- c(found_eogs, sprintf("%s at position %s", 
                                        pattern, 
                                        paste(matches, collapse = ",")))
  }
}

if (length(found_eogs) > 0) {
  cat("\n❌ 发现EOG tokens在输出中:\n")
  for (eog in found_eogs) {
    cat(sprintf("  - %s\n", eog))
  }
  cat("\n🔧 这意味着llama_vocab_is_eog()没有正确识别这些tokens\n")
  
  # 可能的原因分析
  cat("\n📋 可能原因:\n")
  cat("  1. 模型的special_eog_ids集合不包含这些multi-token序列\n")
  cat("  2. llama_vocab_is_eog()只检查单个token，不检查完整的序列\n")
  cat("  3. 这些不是真正的single EOG tokens，而是token序列\n")
  
} else {
  cat("\n✅ 没有发现EOG tokens - llama_vocab_is_eog()可能正常工作\n")
}

# 测试是否EOG检测在某种条件下工作
cat("\n🧪 测试EOS token行为:\n")
simple_result <- generate(model, ctx, tokens_in, max_tokens = 200L)
if (nchar(simple_result) < 200*5) { # 估算平均token长度
  cat("✅ 生成提前停止，可能EOG检测在某些情况下工作\n")
} else {
  cat("❌ 生成没有提前停止，EOG检测可能完全失效\n")
}

rm(model, ctx)
backend_free()
cat("\n✅ 测试完成\n")