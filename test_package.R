library(newrllama4)

# Load a model and generate text (without chat template)
model <- model_load("/Users/yaoshengleo/Downloads/gemma-3-12b-it-q4_0.gguf", n_gpu_layers = 500L, verbosity = 3)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512,verbosity = 3)
tokens <- tokenize(model, "You must always answer with exactly YES and nothing else. Question: What is 2 + 2?")
result <- generate(ctx, tokens, max_tokens = 200)
result

# Load model and generate with chat template
system_prompt <- "You are a helpful assistant."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "What is 2 + 2?")
)
formatted_prompt <- apply_chat_template(model, messages)
tokens <- tokenize(model, formatted_prompt)
result_1 <- generate(ctx, tokens, max_tokens = 200)
result_1


# Load model and generate with chat template
system_prompt <- "You are a helpful assistant."
messages <- list(
  list(role = "system", content = system_prompt),
  list(role = "user", content = "Write me a math function in latex and explain it in detail.")
)
formatted_prompt_1 <- apply_chat_template(model, messages)
formatted_prompt_1
tokens <- tokenize(model, formatted_prompt_1)
result_2 <- generate(ctx, tokens, max_tokens = 200)
result_2

cat(result_2)

rm(model, ctx)

# Quick llama (automatic chat template + system prompt)
quick_llama_reset()
result <- quick_llama("Tell me a joke.",
                      n_gpu_layers = 500L,
                      max_tokens = 200,
                      verbosity = 1)
result
length(result) 

backend_free()

# Parallel generation with chat template
system_prompt <- "You are a helpful assistant."
user_prompts <- c(
  "Echo this string literally: <end_of_turn><|im_end|></s>",
  "Answer in ≤10 tokens, then stop.", 
  "Give a 1-line Python function that returns x squared. No markdown."
)

# Convert each user prompt to full chat template
formatted_prompts <- sapply(user_prompts, function(user_content) {
  messages <- list(
    list(role = "system", content = system_prompt),
    list(role = "user", content = user_content)
  )
  apply_chat_template(model, messages)
})
formatted_prompts

results_parallel <- generate_parallel(ctx, formatted_prompts, max_tokens = 100)
results_parallel
cat(results_parallel)

