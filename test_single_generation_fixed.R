# Fixed Single Sequence Generation EOS Test
# Testing the low-level generate() function with simpler approach

library(newrllama4)

cat("🧪 Testing single sequence generation EOS handling (fixed)...\n")
cat(rep("=", 60), "\n")

# Simple test with just one reliable model
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf"
test_prompt <- "Hello, how are you?"

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

test_single_generation_simple <- function() {
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
  
  cat("📋 Testing single generation with Gemma model\n")
  cat("📁 Model:", basename(model_path), "\n")
  
  tryCatch({
    # Load model
    cat("🔄 Loading model...\n")
    model <- model_load(model_path, n_gpu_layers = 50, verbosity = 2L)  # Less GPU layers
    cat("✅ Model loaded\n")
    
    # Create context
    ctx <- context_create(model, n_ctx = 1024, verbosity = 2L)  # Smaller context
    cat("✅ Context created\n")
    
    # Simple tokenization - no chat template
    tokens <- tokenize(model, test_prompt, add_special = TRUE)
    cat("📝 Tokenized:", length(tokens), "tokens\n")
    
    # Run multiple generations with different parameters
    test_configs <- list(
      list(name = "Conservative", max_tokens = 30, temperature = 0.1, seed = 42),
      list(name = "Balanced", max_tokens = 50, temperature = 0.7, seed = 123),
      list(name = "Creative", max_tokens = 40, temperature = 0.9, seed = 456)
    )
    
    for (i in seq_along(test_configs)) {
      config <- test_configs[[i]]
      cat(sprintf("\n🎯 Test %d: %s settings\n", i, config$name))
      
      start_time <- Sys.time()
      result <- generate(ctx, tokens,
                        max_tokens = config$max_tokens,
                        temperature = config$temperature,
                        seed = config$seed,
                        top_k = 20,
                        top_p = 0.9)
      end_time <- Sys.time()
      
      generation_time <- as.numeric(end_time - start_time)
      cat("⏱️ Generation time:", round(generation_time, 2), "seconds\n")
      cat("📏 Output length:", nchar(result), "characters\n")
      cat("📝 Raw output: '", result, "'\n")
      
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
        cat("❌ EOG LEAK detected! Patterns:", paste(found_patterns, collapse = ", "), "\n")
      } else {
        cat("✅ CLEAN output - no EOG tokens found\n")
      }
      
      # Store result
      results[[config$name]] <- list(
        config = config,
        output = result,
        length = nchar(result),
        clean = !eog_found,
        eog_patterns = found_patterns,
        generation_time = generation_time
      )
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
cat("Starting single generation test...\n")
test_results <- test_single_generation_simple()

# Summary
cat("\n", rep("=", 60), "\n")
cat("📊 SINGLE GENERATION TEST SUMMARY\n")
cat(rep("=", 60), "\n")

if (!is.null(test_results$status)) {
  if (test_results$status == "error") {
    cat("❌ Test failed due to error:", test_results$error, "\n")
  } else if (test_results$status == "model_not_found") {
    cat("⚠️ Test skipped - model file not found\n")
  }
} else {
  # Analyze results
  clean_tests <- sum(sapply(test_results, function(r) r$clean))
  total_tests <- length(test_results)
  
  cat("📈 Clean outputs:", clean_tests, "out of", total_tests, "\n")
  cat("📊 Success rate:", round(100 * clean_tests / total_tests, 1), "%\n")
  
  # Show details
  cat("\n🔍 Test details:\n")
  for (test_name in names(test_results)) {
    result <- test_results[[test_name]]
    status <- if (result$clean) "✅ CLEAN" else "❌ CONTAMINATED"
    cat(sprintf("%s %s: %d chars, %.2fs\n", 
                status, test_name, result$length, result$generation_time))
    
    if (!result$clean) {
      cat(sprintf("   Found EOG patterns: %s\n", paste(result$eog_patterns, collapse = ", ")))
    }
  }
  
  # Final conclusion
  if (clean_tests == total_tests) {
    cat("\n🎉 CONCLUSION: Single generation EOS handling is WORKING perfectly!\n")
  } else if (clean_tests == 0) {
    cat("\n🔴 CONCLUSION: Single generation has SERIOUS EOS leak issues!\n")
  } else {
    cat("\n⚠️ CONCLUSION: Single generation has PARTIAL EOS issues (", 
        total_tests - clean_tests, "out of", total_tests, "tests failed)\n")
  }
}

cat("\nTest completed.\n")