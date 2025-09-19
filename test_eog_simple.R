# Simplified EOG Token Leak Test
# Testing the mixed EOG detection strategy fix

library(newrllama4)

# Test prompt
prompt <- "Tell me a joke"

# EOG patterns to check
eog_patterns <- c(
  "<\\|eot_id\\|>", 
  "<\\|end_header_id\\|>", 
  "<\\|start_header_id\\|>",
  "<end_of_turn>",
  "<start_of_turn>",
  "\\[INST\\]", 
  "\\[/INST\\]"
)

check_eog_leaks <- function(text, model_name) {
  cat("\n=== Testing", model_name, "===\n")
  cat("Output: '", text, "'\n")
  
  leaks_found <- FALSE
  for (pattern in eog_patterns) {
    if (grepl(pattern, text)) {
      cat("❌ Found EOG leak:", pattern, "\n")
      leaks_found <- TRUE
    }
  }
  
  if (!leaks_found) {
    cat("✅ No EOG token leaks detected\n")
  }
  
  return(!leaks_found)
}

cat("Starting EOG token leak test...\n")

# Test 1: Default quick_llama
cat("\n🧪 Test 1: quick_llama (default model)\n")
result1 <- quick_llama(prompt)
clean1 <- check_eog_leaks(result1, "quick_llama")

# Test 2: Gemma model
cat("\n🧪 Test 2: Gemma 3 1B Instruct\n")
tryCatch({
  model2 <- model_load("/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf", n_gpu_layers = -1)
  ctx2 <- context_create(model2, n_ctx = 4096)
  
  # Apply chat template
  chat_msg <- list(list(role = "user", content = prompt))
  templated2 <- apply_chat_template(model2, NULL, chat_msg, add_ass = TRUE)
  
  # Tokenize and generate
  tokens2 <- tokenize(model2, templated2, add_special = TRUE)
  result2 <- generate(ctx2, tokens2, max_tokens = 100, temperature = 0.7, seed = 42)
  
  clean2 <- check_eog_leaks(result2, "Gemma 3-1B")
  
}, error = function(e) {
  cat("❌ Error testing Gemma model:", e$message, "\n")
  clean2 <<- FALSE
})

# Test 3: Llama 3.1 8B Lexi
cat("\n🧪 Test 3: Llama 3.1 8B Lexi Uncensored\n")
tryCatch({
  model3 <- model_load("/Users/yaoshengleo/Desktop/gguf模型/Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf", n_gpu_layers = -1)
  ctx3 <- context_create(model3, n_ctx = 4096)
  
  # Apply chat template
  chat_msg <- list(list(role = "user", content = prompt))
  templated3 <- apply_chat_template(model3, NULL, chat_msg, add_ass = TRUE)
  
  # Tokenize and generate
  tokens3 <- tokenize(model3, templated3, add_special = TRUE)
  result3 <- generate(ctx3, tokens3, max_tokens = 100, temperature = 0.7, seed = 42)
  
  clean3 <- check_eog_leaks(result3, "Llama 3.1 8B Lexi")
  
}, error = function(e) {
  cat("❌ Error testing Llama Lexi model:", e$message, "\n")
  clean3 <<- FALSE
})

# Test 4: Llama 3.1 8B Instruct
cat("\n🧪 Test 4: Llama 3.1 8B Instruct\n")
tryCatch({
  model4 <- model_load("/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf", n_gpu_layers = -1)
  ctx4 <- context_create(model4, n_ctx = 4096)
  
  # Apply chat template
  chat_msg <- list(list(role = "user", content = prompt))
  templated4 <- apply_chat_template(model4, NULL, chat_msg, add_ass = TRUE)
  
  # Tokenize and generate
  tokens4 <- tokenize(model4, templated4, add_special = TRUE)
  result4 <- generate(ctx4, tokens4, max_tokens = 100, temperature = 0.7, seed = 42)
  
  clean4 <- check_eog_leaks(result4, "Llama 3.1 8B Instruct")
  
}, error = function(e) {
  cat("❌ Error testing Llama Instruct model:", e$message, "\n")
  clean4 <<- FALSE
})

# Summary
cat("\n" , strrep("=", 60), "\n")
cat("📊 FINAL SUMMARY\n")
cat(strrep("=", 60), "\n")

clean_models <- c(clean1, exists("clean2") && clean2, exists("clean3") && clean3, exists("clean4") && clean4)
total_tests <- length(clean_models)
passed_tests <- sum(clean_models, na.rm = TRUE)

cat("Tests passed:", passed_tests, "out of", total_tests, "\n")

if (all(clean_models, na.rm = TRUE)) {
  cat("🎉 CONCLUSION: All models PASSED! EOG token leak issue is RESOLVED!\n")
} else {
  cat("⚠️ CONCLUSION: Some models still have EOG token leaks.\n")
}

cat("\nDone.\n")