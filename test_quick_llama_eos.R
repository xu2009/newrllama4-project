# Quick Llama EOS Handling Test
# Testing whether quick_llama properly handles EOS tokens

library(newrllama4)

cat("🧪 Testing quick_llama EOS handling...\n")
cat(rep("=", 60), "\n")

# Test parameters
test_prompts <- c(
  "Tell me a joke",
  "What is the capital of France?",
  "Write a short poem about cats",
  "Explain quantum computing in one sentence"
)

# EOG patterns to check for (using literal strings)
eog_patterns <- c(
  "<|eot_id|>",
  "<|end_header_id|>", 
  "<|start_header_id|>",
  "<end_of_turn>",
  "<start_of_turn>",
  "[INST]",
  "[/INST]",
  "</s>",
  "<s>",
  "<|im_start|>",
  "<|im_end|>"
)

test_quick_llama_eos <- function() {
  results <- list()
  
  for (i in seq_along(test_prompts)) {
    prompt <- test_prompts[i]
    cat("\n--- Test", i, "---\n")
    cat("Prompt:", prompt, "\n")
    
    tryCatch({
      # Generate with quick_llama
      start_time <- Sys.time()
      result <- quick_llama(prompt, max_tokens = 80, temperature = 0.7, seed = 42 + i)
      end_time <- Sys.time()
      
      cat("Generated (", round(as.numeric(end_time - start_time), 2), "s):", nchar(result), "chars\n")
      cat("Output: '", result, "'\n")
      
      # Check for EOG tokens
      eog_found <- FALSE
      for (pattern in eog_patterns) {
        if (grepl(pattern, result, fixed = TRUE)) {
          cat("❌ Found EOG pattern:", pattern, "\n")
          eog_found <- TRUE
        }
      }
      
      if (!eog_found) {
        cat("✅ Clean output - no EOG tokens detected\n")
      }
      
      # Store result
      results[[i]] <- list(
        prompt = prompt,
        output = result,
        clean = !eog_found,
        length = nchar(result)
      )
      
    }, error = function(e) {
      cat("❌ Error:", e$message, "\n")
      results[[i]] <- list(
        prompt = prompt,
        error = e$message,
        clean = FALSE
      )
    })
  }
  
  return(results)
}

# Run the test
test_results <- test_quick_llama_eos()

# Summary
cat("\n", rep("=", 60), "\n")
cat("📊 SUMMARY\n")
cat(rep("=", 60), "\n")

clean_count <- sum(sapply(test_results, function(r) r$clean))
total_tests <- length(test_results)

cat("Tests passed:", clean_count, "out of", total_tests, "\n")

if (clean_count == total_tests) {
  cat("🎉 All quick_llama tests PASSED! EOS handling is working correctly.\n")
} else {
  cat("⚠️ Some quick_llama tests failed. EOS handling needs attention.\n")
}

# Show individual results
for (i in seq_along(test_results)) {
  result <- test_results[[i]]
  status <- if (result$clean) "✅" else "❌"
  cat(sprintf("%s Test %d: %s\n", status, i, result$prompt))
}

cat("\nTest completed.\n")