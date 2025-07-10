# Test v1.0.43: Comprehensive fix for short responses and strange symbols
library(newrllama4)

cat("🔧 Testing v1.0.43 Comprehensive Fixes\n")
cat("=====================================\n")

# Initialize
backend_init()
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.2-1B-Instruct.Q8_0.gguf"
model <- model_load(model_path, n_gpu_layers = 0)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 10)

# Test prompts designed to expose the previous issues
test_prompts <- c(
  "What is artificial intelligence and how does it work in modern applications?",
  "Explain the process of machine learning and its real-world applications.",
  "Describe the benefits and potential risks of AI technology in society."
)

cat("\n📝 Test Prompts:\n")
for (i in seq_along(test_prompts)) {
  cat("  ", i, ": \"", substr(test_prompts[i], 1, 60), "...\"\n", sep = "")
}

# Test with max_tokens=100 (should now generate close to 100 tokens)
cat("\n🚀 Running parallel generation with max_tokens=100...\n")
start_time <- Sys.time()
results <- generate_parallel(ctx, test_prompts, max_tokens = 100)
end_time <- Sys.time()

cat("Generation completed in", round(as.numeric(end_time - start_time), 2), "seconds\n")

# Analyze results
cat("\n📊 RESULTS ANALYSIS:\n")
for (i in seq_along(results)) {
  # Count words and approximate tokens
  words <- length(strsplit(results[i], " ")[[1]])
  chars <- nchar(results[i])
  tokens <- tokenize(model, results[i], add_special = FALSE)
  actual_tokens <- length(tokens)
  
  cat("\n--- Response", i, "---\n")
  cat("Length: ", words, " words, ", actual_tokens, " tokens, ", chars, " chars\n", sep = "")
  cat("Content: \"", substr(results[i], 1, 150), "...\"\n", sep = "")
  
  # Check for artifacts
  has_strange_start <- grepl("^[\\?\\|\\.\\s]", results[i])
  has_extra_newlines <- grepl("\\n\\n", results[i])
  
  if (has_strange_start) {
    cat("⚠️  Warning: Response starts with strange character\n")
  }
  if (has_extra_newlines) {
    cat("⚠️  Warning: Response contains extra newlines\n")
  }
}

# Summary evaluation
word_counts <- sapply(results, function(x) length(strsplit(x, " ")[[1]]))
token_counts <- sapply(results, function(x) length(tokenize(model, x, add_special = FALSE)))

cat("\n🎯 SUMMARY EVALUATION:\n")
cat("Word counts: ", paste(word_counts, collapse = ", "), "\n", sep = "")
cat("Token counts: ", paste(token_counts, collapse = ", "), "\n", sep = "")
cat("Average tokens: ", round(mean(token_counts)), " (target: ~100)\n", sep = "")

# Success criteria check
success_criteria <- list(
  token_length = mean(token_counts) >= 80,  # Should be close to max_tokens=100
  consistency = all(token_counts >= 60),    # All responses substantial
  no_artifacts = !any(sapply(results, function(x) grepl("^[\\?\\|\\.\\s]", x))),
  quality = all(word_counts >= 40)          # All responses meaningful length
)

cat("\n✅ SUCCESS CRITERIA:\n")
cat("Token length (>=80 avg): ", if(success_criteria$token_length) "PASS" else "FAIL", "\n")
cat("Consistency (all >=60): ", if(success_criteria$consistency) "PASS" else "FAIL", "\n") 
cat("No artifacts: ", if(success_criteria$no_artifacts) "PASS" else "FAIL", "\n")
cat("Quality (>=40 words): ", if(success_criteria$quality) "PASS" else "FAIL", "\n")

overall_success <- all(unlist(success_criteria))
cat("\n", if(overall_success) "🎉 OVERALL: SUCCESS!" else "❌ OVERALL: NEEDS MORE WORK", "\n")

if (overall_success) {
  cat("v1.0.43 fixes successfully resolved:\n")
  cat("✅ Short response issue (now generates close to max_tokens)\n")
  cat("✅ Strange symbol artifacts (cleaned up)\n")
  cat("✅ Consistent parallel generation quality\n")
} else {
  cat("Issues remaining - may need additional debugging\n")
}

cat("\nTest completed!\n")