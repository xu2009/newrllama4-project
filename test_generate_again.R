library(newrllama4)

# Load the model - same as quick_llama working version
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  n_gpu_layers = 0,
  verbosity = 1
)

# Create context - same as quick_llama default
ctx <- context_create(model, n_ctx = 2048)

print("Model and context created successfully")

# Test with simple prompt first
simple_prompt <- "Hello, how are you?"
print(paste("Testing with prompt:", simple_prompt))

# Tokenize
tokens <- tokenize(model, simple_prompt)
print(paste("Tokens:", paste(tokens, collapse = ", ")))
print(paste("Token count:", length(tokens)))

# Try generate
print("Calling generate function...")
output_tokens <- generate(ctx, tokens, max_tokens = 10, temperature = 0.7)
print("Generate successful!")

result_text <- detokenize(model, output_tokens)
print(paste("Result:", result_text))