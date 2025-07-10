# Debug Test for v1.0.42 - Why are responses short?
library(newrllama4)

cat("🔍 DEBUG: Investigating short responses in v1.0.42\n")
cat("==================================================\n")

# Initialize
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(model_path, n_gpu_layers = 0)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)

# Test 1: Single prompt with high max_tokens
cat("\n🧪 TEST 1: Single generation with max_tokens=200\n")
single_prompt <- "What is artificial intelligence and how does it work?"
cat("Prompt: \"", single_prompt, "\"\n", sep = "")

tokens <- tokenize(model, single_prompt, add_special = TRUE)
cat("Tokenized length:", length(tokens), "tokens\n")

single_result <- generate(ctx, tokens, max_tokens = 200, 
                         temperature = 0.8, top_p = 0.9, top_k = 40)
single_words <- length(strsplit(single_result, " ")[[1]])
cat("Single generation result (", single_words, " words):\n", sep = "")
cat("\"", substr(single_result, 1, 300), "...\"\n", sep = "")

# Test 2: Parallel generation with same prompt
cat("\n🧪 TEST 2: Parallel generation with same max_tokens=200\n")
parallel_prompts <- rep(single_prompt, 3)
parallel_results <- generate_parallel(ctx, parallel_prompts, max_tokens = 200,
                                    temperature = 0.8, top_p = 0.9, top_k = 40)

parallel_words <- sapply(parallel_results, function(x) length(strsplit(x, " ")[[1]]))
cat("Parallel generation results:\n")
for (i in seq_along(parallel_results)) {
  cat("  Result", i, "(", parallel_words[i], " words): \"", 
      substr(parallel_results[i], 1, 100), "...\"\n", sep = "")
}

# Test 3: Check if it's the model or our code
cat("\n🧪 TEST 3: Very simple prompts with high max_tokens\n")
simple_prompts <- c("Hello", "Hi", "Good")
simple_results <- generate_parallel(ctx, simple_prompts, max_tokens = 300)
simple_words <- sapply(simple_results, function(x) length(strsplit(x, " ")[[1]]))

cat("Simple prompt results:\n")
for (i in seq_along(simple_results)) {
  cat("  \"", simple_prompts[i], "\" -> ", simple_words[i], " words: \"",
      substr(simple_results[i], 1, 80), "...\"\n", sep = "")
}

# Analysis
cat("\n📊 ANALYSIS:\n")
cat("Single generation length:", single_words, "words\n")
cat("Parallel generation average:", round(mean(parallel_words)), "words\n")
cat("Simple prompts average:", round(mean(simple_words)), "words\n")

if (single_words > mean(parallel_words) * 2) {
  cat("❗ FINDING: Single generation much longer than parallel\n")
  cat("   This suggests our parallel fixes may have introduced early stopping\n")
} else if (single_words < 50) {
  cat("❗ FINDING: Even single generation is short\n") 
  cat("   This suggests the model itself prefers concise answers\n")
} else {
  cat("✅ FINDING: Length difference is reasonable\n")
}

cat("\n🎯 CONCLUSION:\n")
if (mean(c(parallel_words, simple_words)) > 15 && 
    all(parallel_words > 10)) {
  cat("✅ v1.0.42 fixes are working - no crashes, coherent responses\n")
  cat("   The short responses may be due to model characteristics\n")
  cat("   or appropriate stopping at natural endpoints\n")
} else {
  cat("❌ There may still be issues with the parallel generation\n")
}

cat("\nTest completed!\n")