library(newrllama4)

# Load model
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf",
  n_gpu_layers = 50,
  verbosity = 1
)

# Create context with conservative settings
ctx <- context_create(model, n_ctx = 2048, n_seq_max = 4)

# Test with just 3 simple prompts
simple_prompts <- c(
  "What is 1+1?",
  "What color is the sky?",
  "What is the capital of France?"
)

cat("Testing parallel generation with", length(simple_prompts), "prompts\n")

# Test parallel generation
result <- generate_parallel(
  context = ctx,
  prompts = simple_prompts,
  max_tokens = 10L,
  temperature = 0.1
)

cat("Results:\n")
for(i in seq_along(result)) {
  cat("Prompt", i, ":", result[[i]], "\n")
}