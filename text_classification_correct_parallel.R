library(dplyr)
library(newrllama4)

# Load bundled sample dataset
data("ag_news_sample", package = "newrllama4")

data_sample <- ag_news_sample %>%
  mutate(LLM_result = NA_character_)

cat("Dataset created with", nrow(data_sample), "observations\n")

response <- quick_llama("What is machine learning in one sentence?")
cat(response)

library(newrllama4)

cache_dir <- get_model_cache_dir()
cached <- list_cached_models()
lock_path <- file.path(cache_dir, "gemma-3-1b-it-q4_0.gguf.lock")

if (file.exists(lock_path)) {
  unlink(lock_path)
  message("清理了残留锁：", lock_path)
} else {
  message("缓存目录里没有锁文件：", cache_dir)
}



# 1. Load the model once
model <- model_load(
  model_path = "https://huggingface.co/MaziyarPanahi/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it.Q4_K_S.gguf",
  n_gpu_layers = 99,
  verbosity = 1
)

# 2. Create a reusable context with n_seq_max smaller than prompt count (testing batch processing)
ctx <- context_create(model, n_ctx = 1024, n_seq_max = 10,verbosity = 1)

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
      "Description: ", substr(data_sample$description[i], 1, 100), "\n\n",
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
    temperature = 0.7,
    top_k = 20L,
    top_p = 0.95,
    repeat_last_n = 32L,
    penalty_repeat = 1.05,
    seed = 1234L,
    progress = TRUE
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
