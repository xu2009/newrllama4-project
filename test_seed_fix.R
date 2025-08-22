# ===================================================================
# 测试种子控制修复效果 - 专门测试KV缓存清除的影响
# ===================================================================
library(newrllama4)

cat("=== 测试种子控制修复效果 ===\n\n")

# 重新编译并加载（如果需要的话，这里先用现有的库测试）
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0)
ctx <- context_create(model, n_ctx = 2048, verbosity = 0)

# 测试参数
test_seed <- 12345L
test_temperature <- 0.7
test_max_tokens <- 30

# 准备相同的prompt
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Write a short story about a cat in exactly 2 sentences.")
)
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)

cat("测试参数: seed =", test_seed, ", temperature =", test_temperature, ", max_tokens =", test_max_tokens, "\n")
cat("Prompt:", "Write a short story about a cat in exactly 2 sentences.\n\n")

# 进行3次独立测试
results <- list()
for(i in 1:3) {
  cat("运行", i, "...")
  result <- generate(ctx, tokens, 
                    max_tokens = test_max_tokens,
                    temperature = test_temperature,
                    seed = test_seed)
  results[[i]] <- result
  cat(" 完成\n")
}

# 检查一致性
all_same <- all(sapply(2:3, function(i) results[[i]] == results[[1]]))

cat("\n结果分析:\n")
cat("一致性:", all_same, "\n")

for(i in 1:3) {
  cat("结果", i, ":", substr(results[[i]], 1, 60), "...\n")
}

if(all_same) {
  cat("\n✅ 种子控制修复成功！KV缓存清除生效。\n")
  cat("完整统一结果:", results[[1]], "\n")
} else {
  cat("\n❌ 种子控制仍然存在问题。\n")
  cat("可能原因:\n")
  cat("1. 修复的代码还未编译生效\n")
  cat("2. 需要重新安装包\n")
  cat("3. 存在其他影响因素\n")
}

# 清理
rm(model, ctx)
backend_free()

cat("\n=== 种子控制修复测试完成 ===\n")