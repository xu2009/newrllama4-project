# Single Sequence Generation EOS Handling Test
# Testing the low-level generate() function EOS behavior

library(newrllama4)

cat("🧪 Testing single sequence generation EOS handling...\n")
cat(rep("=", 60), "\n")

# Test models - will test available ones
test_models <- list(
  "Gemma-3-1B" = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
  "Llama-3.1-8B-Lexi" = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf",
  "Llama-3.1-8B-Instruct" = "/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf"
)

# Test prompt
test_prompt <- "Tell me a joke"

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
  "<s>",
  "<|im_start|>",
  "<|im_end|>"
)

test_single_generation_eos <- function() {
  results <- list()
  
  # Initialize backend
  if (!lib_is_installed()) {
    install_newrllama()
  }
  backend_init()
  
  for (model_name in names(test_models)) {
    model_path <- test_models[[model_name]]
    
    cat("\n", rep("-", 50), "\n")
    cat("📋 Testing:", model_name, "\n")
    cat("📁 Path:", basename(model_path), "\n")
    
    # Check if file exists
    if (!file.exists(model_path)) {
      cat("⚠️ Model file not found, skipping\n")
      results[[model_name]] <- list(status = "file_not_found")
      next
    }
    
    tryCatch({
      # Clean previous state
      backend_free()
      Sys.sleep(1)
      backend_init()
      
      # Load model
      cat("🔄 Loading model...\n")
      model <- model_load(model_path, n_gpu_layers = 999, verbosity = 2L)
      cat("✅ Model loaded successfully\n")
      
      # Create context
      ctx <- context_create(model, n_ctx = 2048, verbosity = 2L)
      cat("✅ Context created\n")
      
      # Apply chat template if available
      cat("📝 Applying chat template...\n")
      chat_messages <- list(list(role = "user", content = test_prompt))
      templated_prompt <- tryCatch({
        apply_chat_template(model, NULL, chat_messages, add_ass = TRUE)
      }, error = function(e) {
        cat("⚠️ Chat template failed, using raw prompt\n")
        test_prompt
      })
      
      # Tokenize
      tokens <- tokenize(model, templated_prompt, add_special = TRUE)
      cat("📝 Tokenized:", length(tokens), "tokens\n")
      
      # Generate with different seeds
      test_seeds <- c(42, 123, 456)
      
      for (i in seq_along(test_seeds)) {
        seed <- test_seeds[i]
        cat(sprintf("🎲 Generation %d with seed %d...\n", i, seed))
        
        start_time <- Sys.time()
        result <- generate(ctx, tokens, 
                          max_tokens = 80, 
                          temperature = 0.7, 
                          seed = seed)
        end_time <- Sys.time()
        
        cat("✅ Generated (", round(as.numeric(end_time - start_time), 2), "s):", nchar(result), "chars\n")
        cat("Raw output:", substr(result, 1, 100), if(nchar(result) > 100) "..." else "", "\n")
        
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
          cat("❌ Found EOG patterns:", paste(found_patterns, collapse = ", "), "\n")
        } else {
          cat("✅ Clean output - no EOG tokens\n")
        }
        
        # Store detailed result
        results[[paste0(model_name, "_seed_", seed)]] <- list(
          model = model_name,
          seed = seed,
          output = result,
          length = nchar(result),
          clean = !eog_found,
          eog_patterns = found_patterns,
          generation_time = as.numeric(end_time - start_time)
        )
      }
      
      # Clean up
      rm(model, ctx)
      backend_free()
      Sys.sleep(1)
      
    }, error = function(e) {
      cat("❌ Test failed:", e$message, "\n")
      results[[model_name]] <- list(
        status = "error",
        error = e$message
      )
      
      # Try to clean up
      tryCatch({ backend_free() }, error = function(e2) {})
    })
  }
  
  return(results)
}

# Run the test
test_results <- test_single_generation_eos()

# Analysis and summary
cat("\n", rep("=", 60), "\n")
cat("📊 SINGLE GENERATION EOS TEST SUMMARY\n")
cat(rep("=", 60), "\n")

successful_tests <- sum(sapply(test_results, function(r) !is.null(r$clean) && r$clean))
total_tests <- sum(sapply(test_results, function(r) !is.null(r$clean)))
failed_tests <- sum(sapply(test_results, function(r) !is.null(r$status) && r$status == "error"))

cat("📈 Successful clean outputs:", successful_tests, "out of", total_tests, "\n")
cat("❌ Failed tests:", failed_tests, "\n")
cat("📊 Success rate:", if(total_tests > 0) round(100 * successful_tests / total_tests, 1) else 0, "%\n")

# Show individual results
cat("\n🔍 Individual test results:\n")
for (test_name in names(test_results)) {
  result <- test_results[[test_name]]
  
  if (!is.null(result$clean)) {
    status <- if (result$clean) "✅ CLEAN" else "❌ CONTAMINATED"
    cat(sprintf("%s %s (seed %d): %d chars\n", 
                status, result$model, result$seed, result$length))
    
    if (!result$clean) {
      cat(sprintf("   Found patterns: %s\n", paste(result$eog_patterns, collapse = ", ")))
    }
  } else if (!is.null(result$status)) {
    cat(sprintf("❌ ERROR %s: %s\n", test_name, result$error))
  }
}

if (successful_tests == total_tests && total_tests > 0) {
  cat("\n🎉 CONCLUSION: Single generation EOS handling is WORKING correctly!\n")
} else if (successful_tests == 0 && total_tests > 0) {
  cat("\n🔴 CONCLUSION: Single generation has SERIOUS EOS issues!\n")
} else if (total_tests > 0) {
  cat("\n⚠️ CONCLUSION: Single generation has PARTIAL EOS issues.\n")
} else {
  cat("\n❓ CONCLUSION: Unable to test due to missing models or errors.\n")
}

cat("\nTest completed.\n")