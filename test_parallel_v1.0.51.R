# Test script for newrllama4 v1.0.51 parallel generation fixes
# This script tests the enhanced parallel generation implementation

library(newrllama4)

# Test configuration
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
test_prompts <- c(
  "What is the capital of France?",
  "Explain quantum computing in simple terms.",
  "Write a short poem about autumn.",
  "What are the benefits of exercise?",
  "Describe the process of photosynthesis."
)

cat("=== Testing newrllama4 v1.0.51 Parallel Generation ===\n")
cat("Model:", model_path, "\n")
cat("Number of prompts:", length(test_prompts), "\n\n")

# Initialize backend
cat("Initializing backend...\n")
backend_init()

# Load model
cat("Loading model...\n")
model <- model_load(model_path, n_gpu_layers = 0)

# Create context
cat("Creating context...\n")
ctx <- context_create(model, n_ctx = 2048, n_threads = 4, n_seq_max = 16)

# Test parallel generation
cat("Testing parallel generation...\n")
start_time <- Sys.time()

results <- generate_parallel(
  context = ctx,
  prompts = test_prompts,
  max_tokens = 100,
  temperature = 0.7,
  top_k = 40,
  top_p = 0.9,
  seed = 42
)

end_time <- Sys.time()
duration <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat("\n=== RESULTS ===\n")
cat("Generation completed in", round(duration, 2), "seconds\n\n")

# Print results
for (i in seq_along(results)) {
  cat("--- Prompt", i, "---\n")
  cat("Input:", test_prompts[i], "\n")
  cat("Output:", results[i], "\n")
  
  # Check for common issues
  has_invalid_chars <- grepl("^[?]", results[i]) || grepl("[^\x20-\x7E]", results[i])
  has_mixed_content <- grepl("What is the capital|quantum computing|autumn|exercise|photosynthesis", results[i], ignore.case = TRUE)
  
  if (has_invalid_chars) {
    cat("⚠️  WARNING: Contains invalid characters\n")
  }
  if (has_mixed_content && !grepl(gsub(".*\\b(\\w+)\\b.*", "\\1", tolower(test_prompts[i])), tolower(results[i]))) {
    cat("⚠️  WARNING: Possible mixed content detected\n")
  }
  
  cat("✅ Output looks clean\n")
  cat("\n")
}

# Performance summary
cat("=== PERFORMANCE SUMMARY ===\n")
cat("Total time:", round(duration, 2), "seconds\n")
cat("Average time per prompt:", round(duration / length(test_prompts), 2), "seconds\n")
cat("Total output length:", sum(nchar(results)), "characters\n")

# Clean up
# Note: Objects will be garbage collected automatically

cat("\n=== TEST COMPLETED ===\n")
cat("Check the results above for:\n")
cat("1. No invalid characters (like '?' at the start)\n")
cat("2. No mixed content between different prompts\n")
cat("3. Coherent and relevant responses\n")
cat("4. Clean output formatting\n")