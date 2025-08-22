# 检查token 106是什么
library(newrllama4)

backend_init()
model <- model_load("/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf", verbosity = 0)

# 测试detokenize token 106
cat("=== Token 106 检查 ===\n")
token_106_text <- detokenize(model, c(106L))
cat("Token 106 对应文本: '", token_106_text, "'\n", sep="")

# 测试token 1 (eos)
token_1_text <- detokenize(model, c(1L))
cat("Token 1 对应文本: '", token_1_text, "'\n", sep="")

# 检查生成的结果是否包含token 106
ctx <- context_create(model, n_ctx = 256, verbosity = 0)
messages <- list(list(role = "user", content = "What is 2+2? Reply with only the number."))
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)

cat("\n=== 生成过程分析 ===\n")
# 尝试较短的生成来捕获问题
result <- generate(ctx, tokens, max_tokens = 3)
cat("3 token生成结果: '", result, "'\n", sep="")

result5 <- generate(ctx, tokens, max_tokens = 5)  
cat("5 token生成结果: '", result5, "'\n", sep="")

result10 <- generate(ctx, tokens, max_tokens = 10)
cat("10 token生成结果: '", result10, "'\n", sep="")

backend_free()