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

# Classify each observation using quick_llama
for (i in 1:nrow(data_sample)) {
  cat("Processing", i, "of", nrow(data_sample), "\n")
  
  tryCatch({
    # Create classification prompt
    prompt <- paste0(
      "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
      "Title: ", data_sample$title[i], "\n",
      "Description: ", substr(data_sample$description[i], 1, 300), "\n\n",
      "Category:"
    )
    
    # Use quick_llama for classification
    result <- quick_llama(
      prompt = prompt,
      model = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
      max_tokens = 5,
      temperature = 0.1,
      n_gpu_layers = 0,
      verbosity = 0
    )
    
    # Clean and store result
    data_sample$LLM_result[i] <- trimws(gsub("\\n.*$", "", result))
    
  }, error = function(e) {
    cat("Error on item", i, ":", e$message, "\n")
    data_sample$LLM_result[i] <- "ERROR"
  })
}

# Display final dataset
print(data_sample)
write.csv(data_sample, "classification_results.csv", row.names = FALSE)
cat("Results saved to classification_results.csv\n")