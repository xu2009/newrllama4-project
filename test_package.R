library(newrllama4)

# Load a model and generate text (without chat template)
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L, verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512,verbosity = 3)
tokens <- tokenize(model, "You must always answer with exactly YES and nothing else. Question: What is 2 + 2?")
result <- generate(ctx, tokens, max_tokens = 200)
result

# Load model and generate with chat template
system_prompt <- "Always answer with exactly YES and nothing else."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "What is 2 + 2?")
)
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)
result_1 <- generate(ctx, tokens, max_tokens = 200)
result_1

rm(model, ctx)

# Quick llama (automatic chat template + system prompt)
quick_llama_reset()
result <- quick_llama("Tell me a joke.",
                      n_gpu_layers = 500L,
                      max_tokens = 200,
                      verbosity = 3)
result
length(result) 

backend_free()

prompts <- c("What is machine learning?", "Tell me a joke.", "Where is the capital of China?")
results <- generate_parallel(ctx, prompts, max_tokens = 50)
results
