library(newrllama4)

# Load the model
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  n_gpu_layers = 50,
  verbosity = 1
)

# Create context
ctx <- context_create(model, n_ctx = 512)

# Test simple tokenization first
print("Testing simple tokenization...")
simple_text <- "Hello world"
tokens <- tokenize(model, simple_text)
print(paste("Simple tokens length:", length(tokens)))

# Test chat template
print("Testing chat template...")
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Hello")
)

formatted_prompt <- apply_chat_template(model, messages)
print("Chat template applied successfully")

# Test tokenization of formatted prompt
chat_tokens <- tokenize(model, formatted_prompt)
print(paste("Chat tokens length:", length(chat_tokens)))

# Test generation
print("Testing generation...")
output_tokens <- generate(ctx, chat_tokens, max_tokens = 3, temperature = 0.7)
result_text <- detokenize(model, output_tokens)
print(paste("Result:", result_text))