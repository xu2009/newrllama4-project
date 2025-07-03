#!/usr/bin/env Rscript

# 测试完整的newrllama4模型推理功能
cat("=== newrllama4 Full Inference Test ===\n\n")

# 1. 加载包
cat("1. Loading newrllama4 package...\n")
suppressPackageStartupMessages({
  library(newrllama4)
})
cat("✅ Package loaded successfully\n\n")

# 2. 确保后端已安装
cat("2. Checking backend installation...\n")
install_newrllama()
cat("✅ Backend ready\n\n")

# 3. 设置模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
cat("3. Using model:", model_path, "\n")

if (!file.exists(model_path)) {
  stop("❌ Model file not found: ", model_path)
}
cat("✅ Model file exists\n\n")

# 4. 加载模型
cat("4. Loading model...\n")
tryCatch({
  model <- model_load(
    model_path = model_path,
    n_gpu_layers = 0L,     # CPU only for testing
    use_mmap = TRUE,
    use_mlock = FALSE
  )
  cat("✅ Model loaded successfully\n")
  cat("   Model type:", class(model), "\n\n")
}, error = function(e) {
  cat("❌ Model loading failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 5. 创建上下文
cat("5. Creating context...\n")
tryCatch({
  context <- context_create(
    model = model,
    n_ctx = 2048L,      # Context length
    n_threads = 4L,     # Thread count
    n_seq_max = 4L      # Max sequences for parallel
  )
  cat("✅ Context created successfully\n")
  cat("   Context type:", class(context), "\n\n")
}, error = function(e) {
  cat("❌ Context creation failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 6. 测试分词功能
cat("6. Testing tokenization...\n")
test_text <- "Hello, how are you today?"
tryCatch({
  tokens <- tokenize(model, test_text, add_special = TRUE)
  cat("✅ Tokenization successful\n")
  cat("   Input text:", test_text, "\n")
  cat("   Tokens:", length(tokens), "tokens:", toString(head(tokens, 10)), "...\n")
  
  # 测试反分词
  detokenized <- detokenize(model, tokens)
  cat("   Detokenized:", detokenized, "\n\n")
}, error = function(e) {
  cat("❌ Tokenization failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 7. 测试聊天模板
cat("7. Testing chat template...\n")
tryCatch({
  messages <- list(
    list(role = "user", content = "What is the capital of France?")
  )
  prompt <- apply_chat_template(
    model = model,
    messages = messages,
    template = NULL,  # Use model's default template
    add_assistant = TRUE
  )
  cat("✅ Chat template applied successfully\n")
  cat("   Formatted prompt:", substr(prompt, 1, 100), "...\n\n")
  
  # 分词提示词用于生成
  prompt_tokens <- tokenize(model, prompt, add_special = FALSE)
  cat("   Prompt tokens:", length(prompt_tokens), "tokens\n\n")
}, error = function(e) {
  cat("❌ Chat template failed:", conditionMessage(e), "\n")
  quit(status = 1)
})

# 8. 测试单序列生成
cat("8. Testing single sequence generation...\n")
tryCatch({
  result <- generate(
    context = context,
    tokens = prompt_tokens,
    max_tokens = 50L,
    top_k = 40L,
    top_p = 0.9,
    temperature = 0.7,
    repeat_last_n = 64L,
    penalty_repeat = 1.1,
    seed = 42L
  )
  cat("✅ Single sequence generation successful\n")
  cat("   Generated text:", result, "\n\n")
}, error = function(e) {
  cat("❌ Single sequence generation failed:", conditionMessage(e), "\n")
  # 非致命错误，继续测试
})

# 9. 测试并行生成
cat("9. Testing parallel generation...\n")
tryCatch({
  prompts <- c(
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming."
  )
  
  results <- generate_parallel(
    context = context,
    prompts = prompts,
    max_tokens = 30L,
    top_k = 40L,
    top_p = 0.9,
    temperature = 0.7,
    repeat_last_n = 64L,
    penalty_repeat = 1.1,
    seed = 42L
  )
  
  cat("✅ Parallel generation successful\n")
  for (i in seq_along(prompts)) {
    cat("   Prompt", i, ":", prompts[i], "\n")
    cat("   Result", i, ":", results[i], "\n\n")
  }
}, error = function(e) {
  cat("❌ Parallel generation failed:", conditionMessage(e), "\n")
  # 非致命错误，继续测试
})

# 10. 测试词汇表功能
cat("10. Testing vocabulary functions...\n")
cat("⏸  Vocabulary functions not yet implemented in R API\n\n")

cat("=== Test Summary ===\n")
cat("✅ Package loading: SUCCESS\n")
cat("✅ Backend installation: SUCCESS\n")
cat("✅ Model loading: SUCCESS\n")
cat("✅ Context creation: SUCCESS\n")
cat("✅ Tokenization: SUCCESS\n")
cat("✅ Chat template: SUCCESS\n")
cat("? Single generation: See above\n")
cat("? Parallel generation: See above\n")
cat("⏸ Vocabulary functions: Not implemented\n")
cat("\n=== Test completed ===\n") 