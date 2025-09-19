library(dplyr)
library(textdata)
library(newrllama4)

# Load data
data <- textdata::dataset_ag_news()

# Randomly sample just 3 from each class (12 total observations)
set.seed(123)
data_sample <- data %>%
  group_by(class) %>%
  slice_sample(n = 3, replace = FALSE) %>%
  ungroup() %>%
  mutate(LLM_result = NA_character_)

# Check data structure
table(data_sample$class)
head(data_sample)

# Load the model
model <- model_load(
  model_path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  n_gpu_layers = 50,
  verbosity = 1
)

# Create context with very small size
ctx <- context_create(model, n_ctx = 512)

# Simple system prompt
system_prompt <- "You are a helpful assistant."

# Classify each observation
for (i in 1:nrow(data_sample)) {
  cat("Processing", i, "of", nrow(data_sample), "\n")
  
  tryCatch({
    # Create user prompt with classification instructions
    user_content <- paste0(
      "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
      "Title: ", data_sample$title[i], "\n",
      "Description: ", substr(data_sample$description[i], 1, 200), "\n\n",
      "Category:"
    )
    
    # Create messages
    messages <- list(
      list(role = "system", content = system_prompt),
      list(role = "user", content = user_content)
    )
    
    # Apply chat template and generate
    formatted_prompt <- apply_chat_template(model, messages)
    tokens <- tokenize(model, formatted_prompt)
    
    # Check if tokens are too long for very small context
    if (length(tokens) > 400) {
      # Further truncate if still too long
      short_description <- substr(data_sample$description[i], 1, 50)
      user_content <- paste0(
        "Classify this news article into exactly one category: World, Sports, Business, or Sci/Tech. Respond with only the category name.\n\n",
        "Title: ", data_sample$title[i], "\n",
        "Description: ", short_description, "\n\n",
        "Category:"
      )
      messages[[2]]$content <- user_content
      formatted_prompt <- apply_chat_template(model, messages)
      tokens <- tokenize(model, formatted_prompt)
    }
    
    output_tokens <- generate(ctx, tokens, max_tokens = 5, temperature = 0.7)
    result_text <- detokenize(model, output_tokens)
    
    # Clean result and store
    data_sample$LLM_result[i] <- trimws(gsub("\\n.*$", "", result_text))
    
  }, error = function(e) {
    cat("Error on item", i, ":", e$message, "\n")
    data_sample$LLM_result[i] <- "ERROR"
  })
}

# Display final dataset
print(data_sample)