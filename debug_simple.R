library(newrllama4)

# Load the model with CPU only
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  n_gpu_layers = 0,
  verbosity = 1
)

# Create context
ctx <- context_create(model, n_ctx = 512)

# Test simple generation
print("Testing CPU-only generation...")
simple_text <- "Hello"
tokens <- tokenize(model, simple_text)
print(paste("Tokens:", paste(tokens, collapse = ", ")))

# Test generation
output_tokens <- generate(ctx, tokens, max_tokens = 3, temperature = 0.7)
result_text <- detokenize(model, output_tokens)
print(paste("Result:", result_text))