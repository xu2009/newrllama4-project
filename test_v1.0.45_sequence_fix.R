# Test v1.0.45: Fix for sequence isolation in parallel generation
library(newrllama4)

cat("🔧 Testing v1.0.45 Sequence Isolation Fix\n")
cat("=========================================\n")

# Initialize
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(model_path, n_gpu_layers = 0)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)

# Test prompts - very different topics to detect cross-contamination
test_prompts <- c(
  "What is artificial intelligence?",
  "How do you cook pasta?", 
  "What is the weather like today?"
)

cat("\n📝 Test Prompts (very different topics):\n")
for (i in seq_along(test_prompts)) {
  cat("  ", i, ": \"", test_prompts[i], "\"\n", sep = "")
}

# Test with moderate max_tokens
cat("\n🚀 Running parallel generation with max_tokens=30...\n")
start_time <- Sys.time()
results <- generate_parallel(ctx, test_prompts, max_tokens = 30)
end_time <- Sys.time()

cat("Generation completed in", round(as.numeric(end_time - start_time), 2), "seconds\n")

# Analyze for cross-contamination
cat("\n📊 SEQUENCE ISOLATION ANALYSIS:\n")
keywords <- list(
  AI = c("AI", "artificial", "intelligence", "machine", "algorithm"),
  cooking = c("pasta", "cook", "boil", "water", "kitchen", "recipe"),
  weather = c("weather", "temperature", "sunny", "rain", "cloud", "forecast")
)

for (i in seq_along(results)) {
  cat("\n--- Response", i, "---\n")
  cat("Content: \"", substr(results[i], 1, 150), "...\"\n", sep = "")
  
  # Check for topic contamination
  response_lower <- tolower(results[i])
  contamination <- list()
  
  for (topic in names(keywords)) {
    matches <- sum(sapply(keywords[[topic]], function(kw) grepl(kw, response_lower)))
    if (matches > 0) {
      contamination[[topic]] <- matches
    }
  }
  
  cat("Topic matches: ")
  if (length(contamination) == 0) {
    cat("None detected")
  } else {
    cat(paste(names(contamination), "=", contamination, collapse = ", "))
  }
  cat("\n")
}

# Expected results check
expected_topics <- c("AI", "cooking", "weather")
cat("\n🎯 ISOLATION SUCCESS CHECK:\n")

isolation_success <- TRUE
for (i in seq_along(results)) {
  response_lower <- tolower(results[i])
  
  # Check if response matches expected topic
  expected_topic <- expected_topics[i]
  expected_matches <- sum(sapply(keywords[[expected_topic]], function(kw) grepl(kw, response_lower)))
  
  # Check for contamination from other topics
  other_topics <- setdiff(names(keywords), expected_topic)
  contamination_count <- 0
  for (topic in other_topics) {
    contamination_count <- contamination_count + sum(sapply(keywords[[topic]], function(kw) grepl(kw, response_lower)))
  }
  
  cat("Response", i, ":")
  if (expected_matches > 0 && contamination_count == 0) {
    cat(" ✅ ISOLATED (topic-appropriate)")
  } else if (contamination_count > 0) {
    cat(" ❌ CONTAMINATED (contains other topics)")
    isolation_success <- FALSE
  } else {
    cat(" ⚠️  UNCLEAR (no clear topic match)")
  }
  cat("\n")
}

cat("\n", if(isolation_success) "🎉 SUCCESS: Sequences are properly isolated!" else "❌ FAILURE: Cross-contamination detected", "\n")

cat("\nTest completed!\n")