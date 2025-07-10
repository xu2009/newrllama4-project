# Test script for quick_llama() function
# This script demonstrates the usage of the new high-level function

# Load the package (assuming it's installed)
# library(newrllama4)

# For testing during development, source the functions directly:
cat("Loading newrllama4 functions...\n")
source("newrllama4/R/zzz.R")     # Load helper functions first
source("newrllama4/R/install.R")
source("newrllama4/R/api.R") 
source("newrllama4/R/quick_llama.R")

# Load required libraries
library(parallel)
library(tools)
library(utils)

cat("Functions loaded successfully!\n\n")

cat("=== Testing quick_llama() function ===\n\n")

# Test 1: Simple single prompt
cat("Test 1: Single prompt\n")
cat("----------------------\n")
tryCatch({
  result <- quick_llama("Hello, how are you?")
  cat("Input: Hello, how are you?\n")
  cat("Output:", result, "\n\n")
}, error = function(e) {
  cat("Error in Test 1:", e$message, "\n\n")
})

# Test 2: Multiple prompts
cat("Test 2: Multiple prompts\n")
cat("------------------------\n")
tryCatch({
  prompts <- c("What is AI?", "Explain machine learning", "Tell me about R programming")
  results <- quick_llama(prompts)
  
  for (i in seq_along(prompts)) {
    cat("Input", i, ":", prompts[i], "\n")
    cat("Output", i, ":", results[[i]], "\n")
  }
  cat("\n")
}, error = function(e) {
  cat("Error in Test 2:", e$message, "\n\n")
})

# Test 3: Custom parameters
cat("Test 3: Custom parameters\n")
cat("-------------------------\n")
tryCatch({
  result <- quick_llama("Tell me a very short story", 
                        temperature = 0.9, 
                        max_tokens = 50)
  cat("Input: Tell me a very short story (temp=0.9, max_tokens=50)\n")
  cat("Output:", result, "\n\n")
}, error = function(e) {
  cat("Error in Test 3:", e$message, "\n\n")
})

# Test 4: Reproducible results with seed
cat("Test 4: Reproducible results\n")
cat("---------------------------\n")
tryCatch({
  result1 <- quick_llama("Generate a random number", seed = 42)
  result2 <- quick_llama("Generate a random number", seed = 42)
  
  cat("Result 1 (seed=42):", result1, "\n")
  cat("Result 2 (seed=42):", result2, "\n")
  cat("Are they identical?", identical(result1, result2), "\n\n")
}, error = function(e) {
  cat("Error in Test 4:", e$message, "\n\n")
})

# Test 5: Reset function
cat("Test 5: Reset function\n")
cat("----------------------\n")
tryCatch({
  quick_llama_reset()
  cat("Reset completed successfully\n\n")
}, error = function(e) {
  cat("Error in Test 5:", e$message, "\n\n")
})

# Test 6: Check function help
cat("Test 6: Function help\n")
cat("--------------------\n")
tryCatch({
  cat("To view help, use: ?quick_llama\n")
  cat("To view help for reset, use: ?quick_llama_reset\n\n")
}, error = function(e) {
  cat("Error in Test 6:", e$message, "\n\n")
})

cat("=== Testing completed ===\n")
cat("Note: Some tests may fail if the backend is not installed.\n")
cat("Run install_newrllama() first if needed.\n")