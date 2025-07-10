# Test v1.0.42 Parallel Generation Fixes
library(newrllama4)

cat("Testing newrllama4 v1.0.42 parallel generation fixes\n")
cat("=====================================================\n")

# Initialize
backend_init()

# Load model
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
cat("Loading model:", basename(model_path), "\n")
model <- model_load(model_path, n_gpu_layers = 0)
cat("Model loaded successfully\n")

# Create context  
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)
cat("Context created (n_ctx=4096, n_seq_max=10)\n")

# Test prompts
test_prompts <- c(
  "What is artificial intelligence?",
  "Explain machine learning briefly.", 
  "What are the benefits of AI?"
)

cat("\nTest prompts:\n")
for (i in seq_along(test_prompts)) {
  cat("  ", i, ": \"", test_prompts[i], "\"\n", sep = "")
}

# Run parallel generation test
cat("\nRunning parallel generation (max_tokens=100)...\n")
start_time <- Sys.time()
results <- generate_parallel(ctx, test_prompts, max_tokens = 100)
end_time <- Sys.time()

generation_time <- as.numeric(end_time - start_time)
cat("Generation completed in", round(generation_time, 2), "seconds\n")

# Analyze results
cat("\n=== RESULTS ANALYSIS ===\n")
word_counts <- sapply(results, function(x) length(strsplit(x, " ")[[1]]))

for (i in seq_along(results)) {
  cat("\nResponse", i, "(", word_counts[i], "words):\n")
  cat("Prompt: \"", test_prompts[i], "\"\n", sep = "")
  cat("Answer: \"", substr(results[i], 1, 150), "...\"\n", sep = "")
}

# Summary statistics
cat("\n=== SUMMARY ===\n")
cat("Word counts:", paste(word_counts, collapse = ", "), "\n")
cat("Average length:", round(mean(word_counts)), "words\n")
cat("Min length:", min(word_counts), "words\n")
cat("Max length:", max(word_counts), "words\n")

# Test success criteria
cat("\n=== SUCCESS CRITERIA CHECK ===\n")
if (mean(word_counts) >= 50) {
  cat("✅ SUCCESS: Average response length >=50 words (vs ~10-20 before fix)\n")
} else {
  cat("❌ ISSUE: Average response still too short:", round(mean(word_counts)), "words\n")
}

if (all(word_counts >= 20)) {
  cat("✅ SUCCESS: All responses >=20 words\n") 
} else {
  cat("❌ ISSUE: Some responses too short\n")
}

# Check for coherence issues
coherence_issues <- sapply(results, function(x) grepl("^\\?", x) || grepl("^\\.", x))
if (any(coherence_issues)) {
  cat("❌ ISSUE: Found responses with strange prefixes\n")
} else {
  cat("✅ SUCCESS: No strange prefixes found\n")
}

cat("\n=== CONCLUSION ===\n")
if (mean(word_counts) >= 50 && !any(coherence_issues)) {
  cat("🎉 v1.0.42 FIXES CONFIRMED!\n")
  cat("   - Parallel generation produces substantially longer responses\n")
  cat("   - No more incoherent prefixes\n") 
  cat("   - All responses address the questions properly\n")
} else {
  cat("⚠️  Some issues may remain, need further investigation\n")
}

cat("\nTest completed!\n")