library(newrllama4)

# Load model
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf",
  n_gpu_layers = 50,
  verbosity = 1
)

# Create context with moderate settings for batch processing
ctx <- context_create(model, n_ctx = 2048, n_seq_max = 8)

# Test with 10 prompts to see the batch processing in action
test_prompts <- paste("What is", 1:10, "plus", 1:10, "?")

cat("Testing parallel generation with", length(test_prompts), "prompts\n")
cat("This should trigger the intelligent workshop batch processing\n")

# Test parallel generation
start_time <- Sys.time()
result <- generate_parallel(
  context = ctx,
  prompts = test_prompts,
  max_tokens = 10L,
  temperature = 0.1
)
end_time <- Sys.time()

cat("Processing time:", as.numeric(end_time - start_time, units = "secs"), "seconds\n")
cat("Results:\n")
for(i in seq_along(result)) {
  cat("Prompt", i, ":", trimws(result[[i]]), "\n")
}