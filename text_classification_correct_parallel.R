library(dplyr)
library(textdata)
library(newrllama4)

# Load data
data <- textdata::dataset_ag_news()

# Randomly sample 25 from each class (100 total observations)
set.seed(123)
data_sample <- data %>%
  group_by(class) %>%
  slice_sample(n = 25, replace = FALSE) %>%
  ungroup() %>%
  mutate(LLM_result = NA_character_)

cat("Dataset created with", nrow(data_sample), "observations\n")

# 1. Load the model once
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf", 
  n_gpu_layers = 50, 
  verbosity = 1
)

# 2. Create a reusable context with more sequences and larger context for parallel processing
# n_seq_max should be at least equal to the number of parallel prompts
# Increase n_ctx to provide more memory per sequence
ctx <- context_create(model, n_ctx = 9000, n_seq_max = 10)

# 3. Prepare all prompts at once
cat("Preparing all prompts for parallel processing...\n")
all_prompts <- character(nrow(data_sample))

for (i in 1:nrow(data_sample)) {
  # Define the conversation
  messages <- list(
    list(role = "system", content = "You are a helpful assistant."),
    list(role = "user", content = paste0(
      "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
      "Title: ", data_sample$title[i], "\n",
      "Description: ", substr(data_sample$description[i], 1, 200), "\n\n",
      "Category:"
    ))
  )
  
  # Apply chat template and store
  all_prompts[i] <- apply_chat_template(model, messages)
}

cat("All prompts prepared. Starting parallel generation...\n")

# Record start time for parallel processing
parallel_start_time <- Sys.time()
cat("Parallel processing started at:", format(parallel_start_time, "%Y-%m-%d %H:%M:%S"), "\n")

# 4. Process all samples in parallel
tryCatch({
  # Use generate_parallel for batch processing
  results <- generate_parallel(
    context = ctx,
    prompts = all_prompts,
    max_tokens = 5L,
    temperature = 0.1,
    top_k = 40L,
    top_p = 0.9,
    repeat_last_n = 64L,
    penalty_repeat = 1.1,
    seed = 1234L
  )

  # Record end time for parallel processing
  parallel_end_time <- Sys.time()
  cat("Parallel processing completed at:", format(parallel_end_time, "%Y-%m-%d %H:%M:%S"), "\n")

  # Calculate and display duration
  parallel_duration <- parallel_end_time - parallel_start_time
  cat("Parallel processing duration:", round(as.numeric(parallel_duration, units = "secs"), 2), "seconds\n")

  # Clean and store results
  data_sample$LLM_result <- sapply(results, function(x) {
    trimws(gsub("\\n.*$", "", x))
  })

  cat("Parallel processing completed successfully!\n")
  
}, error = function(e) {
  cat("Error during parallel processing:", e$message, "\n")
  # Fallback: fill with ERROR
  data_sample$LLM_result <- rep("ERROR", nrow(data_sample))
})

# Display final dataset
print(data_sample)

# Compare with true labels
data_sample <- data_sample %>%
  mutate(correct = ifelse(LLM_result == class, TRUE, FALSE))

# Calculate accuracy
accuracy <- mean(data_sample$correct, na.rm = TRUE)
cat("Classification accuracy:", accuracy, "\n")

# Save results
write.csv(data_sample, "classification_results_correct_parallel.csv", row.names = FALSE)
cat("Results saved to classification_results_correct_parallel.csv\n")

# Performance comparison note
cat("\nPerformance note: This parallel version processes all", nrow(data_sample), 
    "samples simultaneously,\nwhich should be significantly faster than the sequential version.\n")