library(dplyr)
library(textdata)
library(newrllama4)

# Load data and create small test sets
data <- textdata::dataset_ag_news()
set.seed(123)

cat("=== Testing n_seq_max limits ===\n\n")

# Load model once
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf",
  n_gpu_layers = 30,
  verbosity = 1
)

# Test function
test_parallel <- function(n_prompts, n_seq_max, test_name) {
  cat("Test:", test_name, "\n")
  cat("Prompts:", n_prompts, "| n_seq_max:", n_seq_max, "\n")

  # Create test data
  test_data <- data %>%
    group_by(class) %>%
    slice_sample(n = ceiling(n_prompts/4), replace = FALSE) %>%
    ungroup() %>%
    slice_head(n = n_prompts)

  # Create context
  ctx <- context_create(model, n_ctx = 2000, n_seq_max = n_seq_max)

  # Prepare prompts
  prompts <- character(n_prompts)
  for (i in 1:n_prompts) {
    messages <- list(
      list(role = "system", content = "Assistant"),
      list(role = "user", content = paste("Category:", substr(test_data$title[i], 1, 50)))
    )
    prompts[i] <- apply_chat_template(model, messages)
  }

  # Try parallel generation
  tryCatch({
    start_time <- Sys.time()
    results <- generate_parallel(
      context = ctx,
      prompts = prompts,
      max_tokens = 2L,
      temperature = 0.1,
      top_k = 10L,
      top_p = 0.9,
      repeat_last_n = 16L,
      penalty_repeat = 1.1,
      seed = 1234L
    )
    end_time <- Sys.time()
    duration <- as.numeric(end_time - start_time, units = "secs")
    cat("✅ SUCCESS! Duration:", round(duration, 2), "seconds\n")
    cat("Sample results:", paste(sapply(results[1:min(3, length(results))], function(x) trimws(strsplit(x, "\n")[[1]][1])), collapse = ", "), "\n")
  }, error = function(e) {
    cat("❌ FAILED:", e$message, "\n")
  })
  cat("\n")
}

# Test cases
test_parallel(5, 5, "Equal: 5 prompts, 5 sequences")
test_parallel(5, 10, "More sequences: 5 prompts, 10 sequences")
test_parallel(10, 5, "Fewer sequences: 10 prompts, 5 sequences")
test_parallel(3, 3, "Minimal: 3 prompts, 3 sequences")
test_parallel(15, 10, "Exceed capacity: 15 prompts, 10 sequences")

cat("=== Testing completed ===\n")