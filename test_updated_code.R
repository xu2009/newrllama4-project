library(newrllama4)

# ===================================================================
# 基础模型加载（✅ 修复版本）
# ===================================================================
model <- model_load("/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L, verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 64, verbosity = 3)

# ===================================================================
# 1. 无chat template的原始生成（保持不变）
# ===================================================================
tokens <- tokenize(model, "You must always answer with exactly YES and nothing else. Question: What is 2 + 2?")
result <- generate(ctx, tokens, max_tokens = 200)
cat("原始生成结果:\n", result, "\n\n")

# ===================================================================
# 2. 使用自动模型内置template（✅ 修复版本）
# ===================================================================
system_prompt <- "You are a helpful assistant."
prompt <- "What is the square root of 144? Reply with only the number."

messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = prompt)
)
messages

# ✅ 修复：正确的参数顺序
formatted_prompt <- apply_chat_template(model, messages, template = NULL)  # 正确参数顺序
# 或者简化为（默认就是NULL）：
formatted_prompt <- apply_chat_template(model, messages)
cat("生成的Chat Template:\n", formatted_prompt, "\n\n")

tokens <- tokenize(model, formatted_prompt)
result_1 <- generate(ctx, tokens, max_tokens = 200)
result_1

cat("模板生成结果:\n", result_1, "\n\n")

# ===================================================================
# 3. 另一个使用自动template的例子（✅ 修复版本）
# ===================================================================
system_prompt <- "You are a helpful assistant."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "Write me a math function in latex and explain it in detail.")
)

# ✅ 修复：正确的参数顺序
formatted_prompt_1 <- apply_chat_template(model, messages)  # 自动使用Gemma模型内置template
cat("生成的Chat Template:\n")
cat(formatted_prompt_1)
cat("\n\n")

tokens <- tokenize(model, formatted_prompt_1)
result_2 <- generate(ctx, tokens, max_tokens = 200)
result_2
cat("最终结果:\n")
cat(result_2)
cat("\n\n")

# ===================================================================
# 4. Quick llama（保持不变，它内部已经处理template）
# ===================================================================
rm(model, ctx)  # 清理资源

quick_llama_reset()
result <- quick_llama("Write me a math function in latex and explain it in detail.",
                      n_gpu_layers = 500L,
                      max_tokens = 200,
                      verbosity = 3)  # ✅ 修复：使用1L而不是1
result
cat("Quick llama结果:\n", result, "\n\n")

backend_free()

# ===================================================================
# 5. 并行生成优化版本（✅ 修复版本）
# ===================================================================
# 重新加载模型用于并行测试
model <- model_load("/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L, verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 64, verbosity = 3)

system_prompt <- "You are a helpful assistant."
user_prompts <- c(
  "Write the quadratic formula in pure LaTeX (no Markdown fences). Do not explain, only output the formula itself.",
  "Return a JSON object with the population of France (approximate, in millions) and the name of its capital city. Do not add text outside the JSON.",
  "Echo this string exactly and only once: BATCH_TEST_987654321. Do not translate or explain."
)

# ✅ 修复：正确的参数顺序
formatted_prompts <- sapply(user_prompts, function(user_content) {
  messages <- list(
    list(role = "system", content = system_prompt),
    list(role = "user", content = user_content)
  )
  # 正确的参数顺序：model, messages, template
  apply_chat_template(model, messages)
})

formatted_prompts

cat("生成的格式化prompts:\n")
for(i in seq_along(formatted_prompts)) {
  cat(sprintf("=== Prompt %d ===\n", i))
  cat(formatted_prompts[i])
  cat("\n\n")
}

results_parallel <- generate_parallel(ctx, formatted_prompts, max_tokens = 200)

results_parallel

cat("并行生成结果:\n")
for(i in seq_along(results_parallel)) {
  cat(sprintf("=== Result %d ===\n", i))
  cat(results_parallel[i])
  cat("\n\n")
}

# 清理资源
rm(model, ctx)
backend_free()
