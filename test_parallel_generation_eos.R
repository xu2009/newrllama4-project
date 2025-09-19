# Parallel Generation EOS Handling Test
# Testing the generate_parallel() function EOS behavior

library(newrllama4)

cat("🧪 Testing parallel generation EOS handling...\n")
cat(rep("=", 60), "\n")

# Test model
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf"

# Multiple test prompts for parallel generation
test_prompts <- c(
  "Tell me a joke",
  "What is the capital of France?",
  "Write a haiku about rain",
  "Explain machine learning briefly",
  "What colors do you like?",
  "How do birds fly?"
)

# EOG patterns to check for
eog_patterns <- c(
  "<|eot_id|>",
  "<|end_header_id|>",
  "<|start_header_id|>",
  "<end_of_turn>",
  "<start_of_turn>",
  "[INST]",
  "[/INST]",
  "</s>",
  "<s>"
)

test_parallel_generation_eos <- function() {
  results <- list()
  
  # Check if model exists
  if (!file.exists(model_path)) {
    cat("❌ Model file not found:", model_path, "\n")
    return(list(status = "model_not_found"))
  }
  
  # Initialize backend
  if (!lib_is_installed()) {
    install_newrllama()
  }
  backend_init()
  
  cat("📋 Testing parallel generation with Gemma model\n")
  cat("📁 Model:", basename(model_path), "\n")
  cat("📝 Testing", length(test_prompts), "prompts in parallel\n")
  
  tryCatch({
    # Load model
    cat("🔄 Loading model...\n")
    model <- model_load(model_path, n_gpu_layers = 50, verbosity = 2L)
    cat("✅ Model loaded\n")
    
    # Create context with multiple sequences support
    ctx <- context_create(model, n_ctx = 1024, n_seq_max = length(test_prompts) + 2, verbosity = 2L)
    cat("✅ Context created with multi-sequence support\n")
    
    # Run parallel generation with different parameter sets
    test_configs <- list(
      list(
        name = "Conservative", 
        max_tokens = 30, 
        temperature = 0.1, 
        top_k = 10,
        top_p = 0.8,
        seed = 42
      ),
      list(
        name = "Balanced", 
        max_tokens = 50, 
        temperature = 0.7,
        top_k = 40, 
        top_p = 0.9,
        seed = 123
      ),
      list(
        name = "Creative", 
        max_tokens = 40, 
        temperature = 0.9,
        top_k = 60,
        top_p = 0.95,
        seed = 456
      )
    )
    
    for (i in seq_along(test_configs)) {
      config <- test_configs[[i]]
      cat(sprintf("\n🎯 Test %d: %s parallel generation\n", i, config$name))
      
      start_time <- Sys.time()
      
      # Use generate_parallel
      parallel_results <- generate_parallel(
        ctx, 
        test_prompts,
        max_tokens = config$max_tokens,
        temperature = config$temperature,
        top_k = config$top_k,
        top_p = config$top_p,
        seed = config$seed,
        repeat_last_n = 64,
        penalty_repeat = 1.1
      )
      
      end_time <- Sys.time()
      generation_time <- as.numeric(end_time - start_time)
      
      cat("⏱️ Total generation time:", round(generation_time, 2), "seconds\n")
      cat("📊 Generated", length(parallel_results), "responses\n")
      
      # Analyze each result
      clean_count <- 0
      total_chars <- 0
      
      for (j in seq_along(parallel_results)) {
        result <- parallel_results[[j]]
        prompt <- test_prompts[j]
        
        cat(sprintf("\n  📝 Prompt %d: '%s'\n", j, prompt))
        cat(sprintf("     Length: %d chars\n", nchar(result)))
        cat(sprintf("     Output: '%s'\n", substr(result, 1, 80), if(nchar(result) > 80) "..." else ""))
        
        # Check for EOG leaks
        eog_found <- FALSE
        found_patterns <- character(0)
        
        for (pattern in eog_patterns) {
          if (grepl(pattern, result, fixed = TRUE)) {
            eog_found <- TRUE
            found_patterns <- c(found_patterns, pattern)
          }
        }
        
        if (eog_found) {
          cat(sprintf("     ❌ EOG LEAK: %s\n", paste(found_patterns, collapse = ", ")))
        } else {
          cat("     ✅ CLEAN\n")
          clean_count <- clean_count + 1
        }
        
        total_chars <- total_chars + nchar(result)
        
        # Store individual result
        results[[paste0(config$name, "_prompt_", j)]] <- list(
          config = config$name,
          prompt_index = j,
          prompt = prompt,
          output = result,
          length = nchar(result),
          clean = !eog_found,
          eog_patterns = found_patterns
        )
      }
      
      cat(sprintf("\n  📊 Summary for %s config:\n", config$name))
      cat(sprintf("     Clean outputs: %d/%d (%.1f%%)\n", clean_count, length(parallel_results), 
                  100 * clean_count / length(parallel_results)))
      cat(sprintf("     Total characters: %d\n", total_chars))
      cat(sprintf("     Avg chars per response: %.1f\n", total_chars / length(parallel_results)))
      cat(sprintf("     Time per response: %.2fs\n", generation_time / length(parallel_results)))
    }
    
    # Clean up
    rm(model, ctx)
    backend_free()
    
    return(results)
    
  }, error = function(e) {
    cat("❌ Test failed with error:", e$message, "\n")
    
    # Try to clean up
    tryCatch({ backend_free() }, error = function(e2) {})
    
    return(list(status = "error", error = e$message))
  })
}

# Run the test
cat("Starting parallel generation test...\n")
test_results <- test_parallel_generation_eos()

# Final summary
cat("\n", rep("=", 60), "\n")
cat("📊 PARALLEL GENERATION TEST FINAL SUMMARY\n")
cat(rep("=", 60), "\n")

if (!is.null(test_results$status)) {
  if (test_results$status == "error") {
    cat("❌ Test failed due to error:", test_results$error, "\n")
  } else if (test_results$status == "model_not_found") {
    cat("⚠️ Test skipped - model file not found\n")
  }
} else {
  # Overall analysis
  clean_tests <- sum(sapply(test_results, function(r) r$clean))
  total_tests <- length(test_results)
  
  cat("📈 Overall clean outputs:", clean_tests, "out of", total_tests, "\n")
  cat("📊 Overall success rate:", round(100 * clean_tests / total_tests, 1), "%\n")
  
  # Break down by configuration
  configs <- unique(sapply(test_results, function(r) r$config))
  
  cat("\n🔍 Breakdown by configuration:\n")
  for (config in configs) {
    config_results <- test_results[sapply(test_results, function(r) r$config == config)]
    config_clean <- sum(sapply(config_results, function(r) r$clean))
    config_total <- length(config_results)
    
    cat(sprintf("  %s: %d/%d clean (%.1f%%)\n", 
                config, config_clean, config_total, 100 * config_clean / config_total))
  }
  
  # Show failures if any
  failed_tests <- test_results[sapply(test_results, function(r) !r$clean)]
  if (length(failed_tests) > 0) {
    cat("\n❌ Failed tests with EOG leaks:\n")
    for (test_name in names(failed_tests)) {
      result <- failed_tests[[test_name]]
      cat(sprintf("  %s: %s -> patterns: %s\n", 
                  test_name, substr(result$prompt, 1, 30), paste(result$eog_patterns, collapse = ", ")))
    }
  }
  
  # Final conclusion
  if (clean_tests == total_tests) {
    cat("\n🎉 CONCLUSION: Parallel generation EOS handling is WORKING perfectly!\n")
    cat("✅ All", total_tests, "parallel generations were clean with no EOG leaks.\n")
  } else if (clean_tests == 0) {
    cat("\n🔴 CONCLUSION: Parallel generation has SERIOUS EOS leak issues!\n")
    cat("❌ All", total_tests, "parallel generations had EOG leaks.\n")
  } else {
    failure_rate <- round(100 * (total_tests - clean_tests) / total_tests, 1)
    cat(sprintf("\n⚠️ CONCLUSION: Parallel generation has PARTIAL EOS issues!\n"))
    cat(sprintf("📊 %d out of %d tests failed (%.1f%% failure rate)\n", 
                total_tests - clean_tests, total_tests, failure_rate))
  }
}

cat("\nTest completed.\n")