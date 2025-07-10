# Test Script for newrllama4 v1.0.48: Sequence Isolation Fix
# This script tests the complete rewrite of parallel generation to eliminate cross-contamination

library(newrllama4)

cat("=== newrllama4 v1.0.48 Parallel Generation Test ===\n")
cat("Testing: Complete sequence isolation fix\n")
cat("Expected: No cross-contamination between different topics\n\n")

# Initialize backend
backend_init()

# Load model
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-3B-Instruct-uncensored.IQ3_M.gguf"
model <- model_load(model_path, n_gpu_layers = 10)

# Create context with enough sequences for parallel generation
ctx <- context_create(model, n_ctx = 4096, n_threads = 8, n_seq_max = 10)

# Test prompts with VERY different topics to detect contamination
test_prompts <- c(
  "What is artificial intelligence and how does it work?",
  "How do you cook traditional Italian pasta?", 
  "What is the weather like in Tokyo today?",
  "Explain quantum mechanics in simple terms."
)

cat("Test prompts:\n")
for (i in seq_along(test_prompts)) {
  cat(sprintf("%d. %s\n", i, test_prompts[i]))
}
cat("\n")

# Test parallel generation with v1.0.48 fix
cat("Running parallel generation test...\n")
start_time <- Sys.time()

results <- generate_parallel(
  context = ctx,
  prompts = test_prompts,
  max_tokens = 100,
  temperature = 0.8
)

end_time <- Sys.time()
processing_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("Processing completed in %.2f seconds\n\n", processing_time))

# Analyze results for cross-contamination
cat("=== RESULTS ANALYSIS ===\n\n")

# Define keywords for each topic
topic_keywords <- list(
  AI = c("artificial", "intelligence", "algorithm", "machine", "learning", "neural", "data", "computer", "AI"),
  Cooking = c("pasta", "cook", "boil", "sauce", "Italian", "water", "salt", "kitchen", "recipe"),
  Weather = c("weather", "Tokyo", "temperature", "rain", "sunny", "cloudy", "forecast", "climate"),
  Quantum = c("quantum", "physics", "mechanics", "particle", "wave", "electron", "atom", "energy")
)

# Check each response
contamination_found <- FALSE

for (i in seq_along(results)) {
  response <- tolower(results[[i]])
  cat(sprintf("=== Response %d ===\n", i))
  cat(sprintf("Prompt: %s\n", test_prompts[i]))
  cat(sprintf("Response (%d words): %s\n", 
              length(strsplit(results[[i]], "\\s+")[[1]]), 
              results[[i]]))
  
  # Check for topic contamination
  topic_matches <- sapply(names(topic_keywords), function(topic) {
    keywords <- topic_keywords[[topic]]
    matches <- sum(sapply(keywords, function(kw) grepl(kw, response)))
    if (matches > 0) {
      cat(sprintf("  -> Contains %d %s-related terms\n", matches, topic))
    }
    matches
  })
  
  # Expected primary topic for each prompt
  expected_topics <- c("AI", "Cooking", "Weather", "Quantum")
  expected_topic <- expected_topics[i]
  
  # Check if response contains keywords from unexpected topics
  unexpected_matches <- topic_matches[names(topic_matches) != expected_topic]
  if (any(unexpected_matches > 1)) {
    contamination_found <- TRUE
    cat(sprintf("  ⚠️  POTENTIAL CONTAMINATION: Response contains unexpected topic keywords\n"))
  } else {
    cat(sprintf("  ✅ Clean response - no unexpected topic contamination\n"))
  }
  
  cat("\n")
}

# Final assessment
cat("=== FINAL ASSESSMENT ===\n")
if (contamination_found) {
  cat("❌ CROSS-CONTAMINATION DETECTED: Responses show mixing of different topics\n")
  cat("   The sequence isolation fix may need further refinement.\n")
} else {
  cat("✅ SEQUENCE ISOLATION SUCCESS: No cross-contamination detected\n")
  cat("   Each response stayed within its expected topic domain.\n")
}

cat(sprintf("\nPerformance: %.2f seconds for %d prompts (%.2f s/prompt)\n", 
            processing_time, length(test_prompts), processing_time/length(test_prompts)))

# Additional technical metrics
response_lengths <- sapply(results, function(r) length(strsplit(r, "\\s+")[[1]]))
cat(sprintf("Response lengths: %s words\n", paste(response_lengths, collapse=", ")))
cat(sprintf("Average response length: %.1f words\n", mean(response_lengths)))

# Save results for comparison
test_results <- list(
  version = "1.0.48",
  timestamp = Sys.time(),
  prompts = test_prompts,
  responses = results,
  processing_time = processing_time,
  contamination_detected = contamination_found,
  response_lengths = response_lengths
)

saveRDS(test_results, "test_results_v1.0.48.rds")
cat("\nTest results saved to: test_results_v1.0.48.rds\n")

cat("\n=== Test completed ===\n")