# 调试EOG token识别问题
library(newrllama4)

cat("=== 调试EOG Token识别问题 ===\n\n")

backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, verbosity = 1)  # 启用详细日志查看EOG信息

# 检查特殊token
cat("模型特殊token信息:\n")

# 测试tokenize一些特殊字符串
test_strings <- c(
  "<end_of_turn>",
  "<eos>", 
  "</s>",
  "12<end_of_turn>",
  "12"
)

for(test_str in test_strings) {
  tokens <- tokenize(model, test_str, add_special = FALSE)
  cat("字符串 '", test_str, "' -> tokens: ", paste(tokens, collapse=", "), "\n", sep="")
}

cat("\n--- 测试实际生成 ---\n")
ctx <- context_create(model, n_ctx = 256, verbosity = 1)

# 简单测试
messages <- list(
  list(role = "user", content = "What is 2+2? Just the number.")
)
formatted_prompt <- apply_chat_template(model, messages)
cat("格式化prompt:\n", formatted_prompt, "\n\n")

tokens <- tokenize(model, formatted_prompt)
cat("Prompt token数量:", length(tokens), "\n")

# 生成并观察
result <- generate(ctx, tokens, max_tokens = 5)
cat("生成结果: '", result, "'\n", sep="")
cat("结果长度:", nchar(result), "\n")
cat("结果包含<end_of_turn>:", grepl("<end_of_turn>", result), "\n")

# 检查结果的token
if(nchar(result) > 0) {
  result_tokens <- tokenize(model, result, add_special = FALSE)  
  cat("结果对应的tokens:", paste(result_tokens, collapse=", "), "\n")
}

backend_free()
cat("\n=== EOG Token调试完成 ===\n")