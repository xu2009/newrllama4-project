# ====================================================================
# EOG Token Leak Verification Test - Comprehensive Analysis
# ====================================================================
# Purpose: Verify that the mixed EOG detection strategy fixes multi-token
#          sequence leakage in text generation across multiple models
# Date: 2025-08-24
# Version: 1.0.77 verification
# ====================================================================

library(newrllama4)

# Define common EOG token patterns that should NOT appear in output
EOG_PATTERNS <- c(
  "<\\|eot_id\\|>",           # Llama EOT token
  "<\\|end_header_id\\|>",    # Llama end header
  "<\\|start_header_id\\|>",  # Llama start header  
  "<\\|im_start\\|>",         # ChatML start
  "<\\|im_end\\|>",           # ChatML end
  "<end_of_turn>",            # Gemma end of turn
  "<start_of_turn>",          # Gemma start of turn
  "\\[INST\\]",               # Llama instruction start
  "\\[/INST\\]",              # Llama instruction end
  "<s>",                      # BOS token representation
  "</s>"                      # EOS token representation
)

# Test models configuration
TEST_MODELS <- list(
  list(
    name = "Default Quick Llama",
    path = "default",
    use_quick_llama = TRUE
  ),
  list(
    name = "Gemma 3 1B Instruct",
    path = "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf",
    use_quick_llama = FALSE
  ),
  list(
    name = "Llama 3.1 8B Lexi Uncensored", 
    path = "/Users/yaoshengleo/Desktop/gguf模型/Llama-3.1-8B-Lexi-Uncensored_V2_Q4.gguf",
    use_quick_llama = FALSE
  ),
  list(
    name = "Llama 3.1 8B Instruct",
    path = "/Users/yaoshengleo/Desktop/gguf模型/Meta-Llama-3.1-8B-Instruct-Q5_K_L.gguf", 
    use_quick_llama = FALSE
  )
)

# Test prompt
TEST_PROMPT <- "Tell me a joke"

# Detection function for EOG tokens
detect_eog_leaks <- function(text) {
  leaks <- list()
  for (pattern in EOG_PATTERNS) {
    matches <- gregexpr(pattern, text, perl = TRUE)
    if (matches[[1]][1] != -1) {
      match_text <- regmatches(text, matches)[[1]]
      leaks[[length(leaks) + 1]] <- list(
        pattern = pattern,
        matches = match_text,
        count = length(match_text),
        positions = as.vector(matches[[1]])
      )
    }
  }
  return(leaks)
}

# Analysis function for generated text
analyze_generation <- function(text, model_name) {
  cat("\n", strrep("=", 80), "\n")
  cat("MODEL:", model_name, "\n")
  cat(strrep("=", 80), "\n")
  
  # Print raw generated text
  cat("RAW OUTPUT:\n")
  cat("'", text, "'\n\n")
  
  # Character and line analysis
  cat("TEXT ANALYSIS:\n")
  cat("- Length:", nchar(text), "characters\n")
  cat("- Lines:", length(strsplit(text, "\n")[[1]]), "\n")
  
  # EOG leak detection
  leaks <- detect_eog_leaks(text)
  
  if (length(leaks) == 0) {
    cat("✅ EOG LEAK STATUS: CLEAN - No EOG tokens detected\n")
    return(list(status = "CLEAN", leaks = list(), text = text))
  } else {
    cat("❌ EOG LEAK STATUS: CONTAMINATED - Found", length(leaks), "leak pattern(s)\n")
    
    for (i in seq_along(leaks)) {
      leak <- leaks[[i]]
      cat(sprintf("   Leak %d: Pattern '%s' found %d time(s)\n", 
                  i, leak$pattern, leak$count))
      cat(sprintf("   Matches: %s\n", paste(leak$matches, collapse = ", ")))
      cat(sprintf("   Positions: %s\n", paste(leak$positions, collapse = ", ")))
    }
    return(list(status = "CONTAMINATED", leaks = leaks, text = text))
  }
}

# Advanced tokenization analysis (if model is loaded)
analyze_raw_tokens <- function(model, text, model_name) {
  if (is.null(model)) return(NULL)
  
  cat("\n", "TOKENIZATION ANALYSIS:", "\n")
  cat(strrep("-", 50), "\n")
  
  tryCatch({
    # Tokenize the generated text
    tokens <- tokenize(model, text, add_special = FALSE)
    cat("Generated tokens:", paste(tokens, collapse = ", "), "\n")
    
    # Check for known problematic token sequences
    problematic_sequences <- list(
      "eot_id" = c(27, 91, 68, 354, 851, 91, 29),  # <|eot_id|>
      "end_header" = c(27, 91, 416, 8932, 851, 91, 29) # <|end_header_id|>
    )
    
    for (seq_name in names(problematic_sequences)) {
      seq_tokens <- problematic_sequences[[seq_name]]
      if (any(sapply(1:(length(tokens) - length(seq_tokens) + 1), function(i) {
        all(tokens[i:(i + length(seq_tokens) - 1)] == seq_tokens)
      }))) {
        cat("❌ Found problematic sequence:", seq_name, "\n")
      }
    }
    
    # Check individual EOG tokens
    eog_tokens_found <- sapply(tokens, function(t) token_is_eog(model, t))
    if (any(eog_tokens_found)) {
      cat("❌ Found", sum(eog_tokens_found), "individual EOG tokens at positions:", 
          which(eog_tokens_found), "\n")
    } else {
      cat("✅ No individual EOG tokens found\n")
    }
    
  }, error = function(e) {
    cat("⚠️ Tokenization analysis failed:", e$message, "\n")
  })
}

# Main testing function
run_comprehensive_eog_test <- function() {
  cat("╔══════════════════════════════════════════════════════════════════╗\n")
  cat("║                    EOG TOKEN LEAK TEST SUITE                    ║\n")  
  cat("║                        Version 1.0.77                           ║\n")
  cat("╚══════════════════════════════════════════════════════════════════╝\n\n")
  
  # Initialize results
  test_results <- list()
  
  cat("🔧 INITIALIZATION\n")
  cat("- Checking backend library status...\n")
  if (!lib_is_installed()) {
    cat("❌ Backend library not installed. Installing now...\n")
    install_newrllama()
  } else {
    cat("✅ Backend library ready\n")
  }
  
  # Test each model
  for (i in seq_along(TEST_MODELS)) {
    model_config <- TEST_MODELS[[i]]
    model_name <- model_config$name
    
    cat("\n", "🧪 TESTING MODEL", i, "of", length(TEST_MODELS), "\n")
    
    tryCatch({
      if (model_config$use_quick_llama) {
        # Test with quick_llama (uses default model)
        cat("Using quick_llama() with default model...\n")
        response <- quick_llama(TEST_PROMPT)
        result <- analyze_generation(response, model_name)
        result$method <- "quick_llama"
        result$model_path <- "default"
        
      } else {
        # Test with explicit model loading
        cat("Loading model:", model_config$path, "\n")
        
        if (!file.exists(model_config$path)) {
          cat("❌ Model file not found:", model_config$path, "\n")
          next
        }
        
        # Load model with GPU acceleration
        cat("Loading with GPU acceleration...\n")
        model <- model_load(model_config$path, n_gpu_layers = -1)
        
        # Create context  
        ctx <- context_create(model, n_ctx = 4096)
        
        # Generate response
        cat("Generating response...\n")
        # Apply chat template first
        chat_messages <- list(list(role = "user", content = TEST_PROMPT))
        templated_prompt <- apply_chat_template(model, NULL, chat_messages, add_ass = TRUE)
        
        # Tokenize prompt
        tokens <- tokenize(model, templated_prompt, add_special = TRUE)
        
        # Generate
        response <- generate(ctx, tokens, max_tokens = 150, temperature = 0.7, seed = 12345)
        
        result <- analyze_generation(response, model_name)
        result$method <- "explicit_loading"
        result$model_path <- model_config$path
        
        # Perform tokenization analysis
        analyze_raw_tokens(model, response, model_name)
      }
      
      test_results[[model_name]] <- result
      
    }, error = function(e) {
      cat("❌ Test failed for", model_name, ":", e$message, "\n")
      test_results[[model_name]] <- list(
        status = "ERROR", 
        error = e$message,
        model_path = model_config$path
      )
    })
    
    # Add separation between tests
    cat("\n", strrep("─", 80), "\n")
  }
  
  # Summary analysis
  cat("\n")
  cat("╔══════════════════════════════════════════════════════════════════╗\n")
  cat("║                           FINAL SUMMARY                         ║\n")
  cat("╚══════════════════════════════════════════════════════════════════╝\n\n")
  
  clean_count <- sum(sapply(test_results, function(r) r$status == "CLEAN"))
  contaminated_count <- sum(sapply(test_results, function(r) r$status == "CONTAMINATED"))
  error_count <- sum(sapply(test_results, function(r) r$status == "ERROR"))
  
  cat("📊 OVERALL RESULTS:\n")
  cat("   ✅ Clean outputs:", clean_count, "\n")
  cat("   ❌ Contaminated outputs:", contaminated_count, "\n")
  cat("   ⚠️ Errors:", error_count, "\n")
  cat("   📈 Success rate:", round(100 * clean_count / length(test_results), 1), "%\n\n")
  
  if (clean_count == length(test_results)) {
    cat("🎉 CONCLUSION: All tests PASSED! EOG token leak issue appears to be RESOLVED.\n")
  } else if (contaminated_count > 0) {
    cat("⚠️ CONCLUSION: EOG token leaks still present in", contaminated_count, "model(s). Issue NOT fully resolved.\n")
  } else {
    cat("❓ CONCLUSION: Unable to determine due to errors. Manual investigation needed.\n")
  }
  
  # Return results for further analysis
  return(invisible(test_results))
}

# Execute the comprehensive test
cat("Starting comprehensive EOG token leak verification...\n\n")
test_results <- run_comprehensive_eog_test()