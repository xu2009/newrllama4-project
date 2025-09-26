library(dplyr)
library(newrllama4)

# Load bundled sample dataset
data("ag_news_sample", package = "newrllama4")

data_sample <- ag_news_sample %>%
  mutate(LLM_result = NA_character_)

cat("Dataset created with", nrow(data_sample), "observations\n")

# 1. Load the model once
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf", 
  n_gpu_layers = 50, 
  verbosity = 3
)

# 2. Create a reusable context
ctx_1 <- context_create(model, n_ctx = 1024)

# Record start time for single sequence processing
single_start_time <- Sys.time()
cat("Single sequence processing started at:", format(single_start_time, "%Y-%m-%d %H:%M:%S"), "\n")

# Process each observation
for (i in 1:nrow(data_sample)) {
  cat("Processing", i, "of", nrow(data_sample), "\n")
  
  tryCatch({
    # 3. Define the conversation
    messages <- list(
      list(role = "system", content = "You are a helpful assistant."),
      list(role = "user", content = paste0(
        "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
        "Title: ", data_sample$title[i], "\n",
        "Description: ", substr(data_sample$description[i], 1, 200), "\n\n",
        "Category:"
      ))
    )
    
    # 4. Apply chat template
    formatted_prompt <- apply_chat_template(model, messages)
    
    # 5. Tokenize and generate
    tokens <- tokenize(model, formatted_prompt)
    output_tokens <- generate(ctx_1, tokens, max_tokens = 5, temperature = 0.1)
    
    # Store the result (output_tokens is already text)
    data_sample$LLM_result[i] <- trimws(gsub("\\n.*$", "", output_tokens))
    
  }, error = function(e) {
    cat("Error on item", i, ":", e$message, "\n")
    data_sample$LLM_result[i] <- "ERROR"
  })
}

# Record end time for single sequence processing
single_end_time <- Sys.time()
cat("Single sequence processing completed at:", format(single_end_time, "%Y-%m-%d %H:%M:%S"), "\n")

# Calculate and display duration
single_duration <- single_end_time - single_start_time
cat("Single sequence processing duration:", round(as.numeric(single_duration, units = "secs"), 2), "seconds\n")

# Display final dataset
print(data_sample)

# Compare with true labels
data_sample <- data_sample %>%
  mutate(correct = ifelse(LLM_result == class, TRUE, FALSE))

# Calculate accuracy
accuracy <- mean(data_sample$correct, na.rm = TRUE)
accuracy

write.csv(data_sample, "classification_results_correct.csv", row.names = FALSE)
cat("Results saved to classification_results_correct.csv\n")
