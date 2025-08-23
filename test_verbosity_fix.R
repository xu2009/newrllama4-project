#!/usr/bin/env Rscript

# Test script to verify verbosity functionality
library(newrllama4)

cat("Testing verbosity levels with model loading...\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  cat("Model file not found at:", model_path, "\n")
  cat("Please update the path in the script or use a different model file.\n")
  quit(status = 1)
}

cat("\n=== Testing verbosity = 0 (all messages) ===\n")
model0 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0L)

cat("\n=== Testing verbosity = 1 (important messages, default) ===\n")  
model1 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 1L)

cat("\n=== Testing verbosity = 2 (warnings and errors) ===\n")
model2 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 2L)

cat("\n=== Testing verbosity = 3 (errors only) ===\n")
model3 <- model_load(model_path, n_gpu_layers = 500L, verbosity = 3L)

cat("\n=== Testing context creation with different verbosity ===\n")

cat("\nCreating context with verbosity = 0:\n")
ctx0 <- context_create(model0, n_ctx = 512L, verbosity = 0L)

cat("\nCreating context with verbosity = 3 (should be very quiet):\n")
ctx3 <- context_create(model3, n_ctx = 512L, verbosity = 3L)

cat("\nVerbosity test completed!\n")