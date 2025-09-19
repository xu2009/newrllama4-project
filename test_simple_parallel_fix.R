library(dplyr)
library(textdata)
library(newrllama4)

# Load data
data <- textdata::dataset_ag_news()

# Sample only 4 examples (1 from each class) for testing
set.seed(123)
data_sample <- data %>%
  group_by(class) %>%
  slice_sample(n = 1, replace = FALSE) %>%
  ungroup() %>%
  mutate(LLM_result = NA_character_)

cat("Dataset created with", nrow(data_sample), "observations\n")

# Load the model
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf",
  n_gpu_layers = 30,  # Reduced GPU layers
  verbosity = 1
)

# Create context with conservative settings
ctx <- context_create(model, n_ctx = 4000, n_seq_max = 4)  # Match the number of samples

# Test single generation first
cat("Testing single generation first...\n")
test_messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Classify this as: World, Sports, Business, or Sci/Tech. Answer with one word only.\n\nTitle: Apple releases new iPhone\n\nCategory:")
)
test_prompt <- apply_chat_template(model, test_messages)
test_tokens <- tokenize(model, test_prompt)
result <- generate(ctx, test_tokens, max_tokens = 3, temperature = 0.1)
cat("Single generation test result:", result, "\n")

# Prepare prompts for parallel
cat("Preparing prompts for parallel processing...\n")
all_prompts <- character(nrow(data_sample))

for (i in 1:nrow(data_sample)) {
  messages <- list(
    list(role = "system", content = "You are a helpful assistant."),
    list(role = "user", content = paste0(
      "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
      "Title: ", data_sample$title[i], "\n",
      "Description: ", substr(data_sample$description[i], 1, 100), "\n\n",
      "Category:"
    ))
  )
  all_prompts[i] <- apply_chat_template(model, messages)
}

cat("Starting parallel generation with", length(all_prompts), "prompts...\n")

# Try parallel generation
tryCatch({
  results <- generate_parallel(
    context = ctx,
    prompts = all_prompts,
    max_tokens = 3L,  # Very short response
    temperature = 0.1,
    top_k = 20L,
    top_p = 0.95,
    repeat_last_n = 32L,
    penalty_repeat = 1.05,
    seed = 1234L
  )

  data_sample$LLM_result <- sapply(results, function(x) {
    trimws(gsub("\\n.*$", "", x))
  })

  cat("Parallel processing completed successfully!\n")
  print(data_sample)

}, error = function(e) {
  cat("Error during parallel processing:", e$message, "\n")
  data_sample$LLM_result <- rep("ERROR", nrow(data_sample))
  print(data_sample)
})