library(newrllama4)

# ===================================================================
# 基础模型加载（这部分保持不变）
# ===================================================================
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = "auto", verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 64, verbosity = 2)

# ===================================================================
# 1. 无chat template的原始生成（保持不变）
# ===================================================================
tokens <- tokenize(model, "You must always answer with exactly YES and nothing else. Question: What is 2 + 2?")
result <- generate(ctx, tokens, max_tokens = 200)
result

# ===================================================================
# 2. 使用自动模型内置template（推荐方式）
# ===================================================================
system_prompt <- "You are a helpful assistant."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "What is 2 + 2?")
)

# 🔧 修改：使用NULL让系统自动使用模型内置template
formatted_prompt <- apply_chat_template(model, messages, tmpl = NULL)  # 显式使用NULL
# 或者简化为（默认就是NULL）：
formatted_prompt <- apply_chat_template(model, messages)
formatted_prompt

tokens <- tokenize(model, formatted_prompt)
result_1 <- generate(ctx, tokens, max_tokens = 200)
result_1

# ===================================================================
# 3. 另一个使用自动template的例子
# ===================================================================
system_prompt <- "You are a helpful assistant."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "Write me a math function in latex and explain it in detail.")
)

# 🔧 修改：利用自动模型内置template
formatted_prompt_1 <- apply_chat_template(model, messages)  # 自动使用Gemma模型内置template
cat("生成的Chat Template:\n")
cat(formatted_prompt_1)
cat("\n\n")

tokens <- tokenize(model, formatted_prompt_1)
result_2 <- generate(ctx, tokens, max_tokens = 200)
result_2

cat("最终结果:\n")
cat(result_2)

# ===================================================================
# 4. Quick llama（保持不变，它内部已经处理template）
# ===================================================================
rm(model, ctx)  # 清理资源

quick_llama_reset()
result <- quick_llama("Tell me a joke.",
                      n_gpu_layers = "auto",
                      max_tokens = 200,
                      verbosity = 1)
result
cat(result)
length(result)

backend_free()

# ===================================================================
# 5. 并行生成优化版本
# ===================================================================
# 重新加载模型用于并行测试
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = "auto", verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 1)

system_prompt <- "You are a helpful assistant."
user_prompts <- c(
  "Echo this string literally: <end_of_turn><|im_end|></s>",
  "Answer in ≤10 tokens, then stop.",
  "Give a 1-line Python function that returns x squared. No markdown."
)

# 🔧 修改：使用自动模型内置template
formatted_prompts <- sapply(user_prompts, function(user_content) {
  messages <- list(
    list(role = "system", content = system_prompt),
    list(role = "user", content = user_content)
  )
  # 自动使用模型内置template
  apply_chat_template(model, messages)
})

cat("生成的格式化prompts:\n")
for(i in seq_along(formatted_prompts)) {
  cat(sprintf("=== Prompt %d ===\n", i))
  cat(formatted_prompts[i])
  cat("\n\n")
}

results_parallel <- generate_parallel(ctx, formatted_prompts, max_tokens = 100)
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
