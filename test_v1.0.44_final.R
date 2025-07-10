# Final Test for v1.0.44: Critical fixes for token generation
library(newrllama4)

cat("🔧 Testing v1.0.44 Critical Fixes\n")
cat("=================================\n")

# Initialize
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(model_path, n_gpu_layers = 0)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)

# Test prompts
test_prompts <- c(
  "What is artificial intelligence?",
  "Explain machine learning briefly.",
  "What are the benefits of AI?"
)

cat("\n📝 Test Prompts:\n")
for (i in seq_along(test_prompts)) {
  cat("  ", i, ": \"", test_prompts[i], "\"\n", sep = "")
}

# Test with max_tokens=50 (more reasonable for testing)
cat("\n🚀 Running parallel generation with max_tokens=50...\n")
start_time <- Sys.time()
results <- generate_parallel(ctx, test_prompts, max_tokens = 50)
end_time <- Sys.time()

cat("Generation completed in", round(as.numeric(end_time - start_time), 2), "seconds\n")

# Analyze results
cat("\n📊 RESULTS ANALYSIS:\n")
for (i in seq_along(results)) {
  words <- length(strsplit(results[i], " ")[[1]])
  chars <- nchar(results[i])
  
  cat("\n--- Response", i, "---\n")
  cat("Length: ", words, " words, ", chars, " chars\n", sep = "")
  cat("Content: \"", results[i], "\"\n", sep = "")
}

# Summary
word_counts <- sapply(results, function(x) length(strsplit(x, " ")[[1]]))

cat("\n🎯 SUMMARY:\n")
cat("Word counts: ", paste(word_counts, collapse = ", "), "\n", sep = "")
cat("Average: ", round(mean(word_counts)), " words\n", sep = "")
cat("All responses > 10 words: ", all(word_counts > 10), "\n", sep = "")
cat("Average > 25 words: ", mean(word_counts) > 25, "\n", sep = "")

if (all(word_counts > 15) && mean(word_counts) > 25) {
  cat("\n🎉 SUCCESS! v1.0.44 fixes are working:\n")
  cat("  - Responses are substantial (>15 words each)\n")
  cat("  - Average length is good (>25 words)\n")
  cat("  - No empty or very short responses\n")
} else {
  cat("\n❌ Still issues with response length\n")
}

cat("\nTest completed!\n")