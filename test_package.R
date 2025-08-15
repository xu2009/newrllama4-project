library(newrllama4)

# Load a model and generate text
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L)
ctx <- context_create(model, n_ctx = 1024, n_seq_max = 6)
tokens <- tokenize(model, "Write me a r function and expalin it.")
result <- generate(ctx, tokens, max_tokens = 100)
result

prompts <- c("What is machine learning?", "Tell me a joke.", "Where is the capital of China?")
results <- generate_parallel(ctx, prompts, max_tokens = 50)
results
