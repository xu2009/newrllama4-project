# ===================================================================
# 种子控制交叉验证测试 - 确认不一致性问题
# ===================================================================
library(newrllama4)

cat("=== 种子控制交叉验证测试 ===\n\n")

# 测试参数
test_seed <- 42L
test_temperature <- 0.7
test_max_tokens <- 30
test_prompt <- "What is 2+2? Answer with just the number."

# ===================================================================
# 1. 测试单序列generate函数的种子一致性
# ===================================================================
cat("--- 测试1: 单序列generate函数种子一致性 ---\n")

# 准备模型和上下文
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0)
ctx <- context_create(model, n_ctx = 2048, verbosity = 0)

# 准备相同的prompt
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = test_prompt)
)
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)

cat("使用参数: seed =", test_seed, ", temperature =", test_temperature, "\n")

# 多次运行单序列函数
generate_results <- list()
for(i in 1:3) {
  cat("  单序列运行", i, "...")
  result <- generate(ctx, tokens, 
                    max_tokens = test_max_tokens,
                    temperature = test_temperature,
                    seed = test_seed)
  generate_results[[i]] <- result
  cat(" 完成\n")
}

# 检查一致性
generate_consistent <- all(sapply(2:3, function(i) generate_results[[i]] == generate_results[[1]]))
cat("单序列generate一致性:", generate_consistent, "\n")
for(i in 1:3) {
  cat("  运行", i, ":", substr(generate_results[[i]], 1, 30), "...\n")
}

# 清理资源，准备下一个测试
rm(model, ctx)
backend_free()

# ===================================================================
# 2. 测试quick_llama函数的种子一致性
# ===================================================================
cat("\n--- 测试2: quick_llama函数种子一致性 ---\n")

cat("使用参数: seed =", test_seed, ", temperature =", test_temperature, "\n")

# 多次运行quick_llama函数
quick_llama_results <- list()
for(i in 1:3) {
  cat("  quick_llama运行", i, "...")
  quick_llama_reset()  # 重置状态确保独立性
  result <- quick_llama(test_prompt,
                       n_gpu_layers = 500L,
                       max_tokens = test_max_tokens,
                       temperature = test_temperature,
                       seed = test_seed,
                       verbosity = 3L)  # 减少输出噪音
  quick_llama_results[[i]] <- result
  cat(" 完成\n")
}

# 检查一致性
quick_llama_consistent <- all(sapply(2:3, function(i) quick_llama_results[[i]] == quick_llama_results[[1]]))
cat("quick_llama一致性:", quick_llama_consistent, "\n")
for(i in 1:3) {
  cat("  运行", i, ":", substr(quick_llama_results[[i]], 1, 30), "...\n")
}

# ===================================================================
# 3. 对比相同种子下的结果差异
# ===================================================================
cat("\n--- 测试3: 对比相同种子下的结果差异 ---\n")

# 重新运行单次进行对比
cat("重新运行单次对比测试...\n")

# 单序列函数
backend_init()
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0)
ctx <- context_create(model, n_ctx = 2048, verbosity = 0)
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)

generate_result_final <- generate(ctx, tokens, 
                                max_tokens = test_max_tokens,
                                temperature = test_temperature,
                                seed = test_seed)

rm(model, ctx)
backend_free()

# quick_llama函数
quick_llama_reset()
quick_llama_result_final <- quick_llama(test_prompt,
                                      n_gpu_layers = 500L,
                                      max_tokens = test_max_tokens,
                                      temperature = test_temperature,
                                      seed = test_seed,
                                      verbosity = 3L)

cat("单序列generate结果:", substr(generate_result_final, 1, 50), "...\n")
cat("quick_llama结果:", substr(quick_llama_result_final, 1, 50), "...\n")
cat("两者结果相同:", generate_result_final == quick_llama_result_final, "\n")

# ===================================================================
# 4. 分析可能的原因
# ===================================================================
cat("\n--- 分析结果 ---\n")

if (!generate_consistent) {
  cat("❌ 问题确认: 单序列generate函数种子控制失效\n")
  cat("   这表明C++层面的种子设置没有正确生效\n")
}

if (quick_llama_consistent) {
  cat("✅ quick_llama函数种子控制正常工作\n")
  cat("   但这可能是因为使用了不同的默认模型路径\n")
}

if (generate_result_final != quick_llama_result_final) {
  cat("⚠️  相同种子下，两个函数产生不同结果\n")
  cat("   可能原因:\n")
  cat("   1. quick_llama使用了不同的默认模型\n")
  cat("   2. quick_llama有额外的文本清理步骤\n")
  cat("   3. 种子传递机制不同\n")
}

cat("\n=== 交叉验证测试完成 ===\n")