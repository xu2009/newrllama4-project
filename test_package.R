library(newrllama4)

# Load a model and generate text
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L,verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512)
tokens <- tokenize(model, "Hello, how are you doing today?")
result <- generate(ctx, tokens, max_tokens = 300)
result

ctx_1 <- context_create(model, n_ctx = 4096, n_seq_max = 512)
token <- tokenize(model, "Hello, how are you doing today?")
result_1 <- generate(ctx_1, token, max_tokens = 300)
result_1

length(result) 

prompts <- c("What is machine learning?", "Tell me a joke.", "Where is the capital of China?")
results <- generate_parallel(ctx, prompts, max_tokens = 50)
results
