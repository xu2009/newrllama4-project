# Bottom-Level EOG Mechanism Test
# Testing raw C++ level EOG prevention WITHOUT R-level post-processing
# This test verifies if the underlying C++ mechanisms are effectively preventing EOG leaks

library(newrllama4)

cat("🔬 Testing BOTTOM-LEVEL EOG prevention mechanisms...\\n")
cat(rep("=", 70), "\\n")

# Test model
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-1b-it.Q8_0.gguf"

# EOG patterns that should be prevented at the C++ level
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

# Test prompts designed to potentially trigger EOG sequences
test_prompts <- c(
  "Hello, how are you today?",
  "What is the meaning of life?",
  "Tell me a joke about programming",
  "Explain quantum physics briefly",
  "Write a haiku about nature",
  "How do computers work?",
  "What's your favorite color?",
  "Describe the weather"
)

test_bottom_level_eog <- function() {
  results <- list()
  
  # Check if model exists
  if (!file.exists(model_path)) {
    cat("❌ Model file not found:", model_path, "\\n")
    return(list(status = "model_not_found"))
  }
  
  # Initialize backend
  if (!lib_is_installed()) {
    install_newrllama()
  }
  backend_init()
  
  cat("🎯 Testing with Gemma model (RAW output, NO post-processing)\\n")
  cat("📁 Model:", basename(model_path), "\\n")
  cat("📝 Testing", length(test_prompts), "different prompts\\n")
  cat("⚠️  IMPORTANT: This test bypasses ALL R-level cleaning\\n")
  
  tryCatch({
    # Load model
    cat("\\n🔄 Loading model...\\n")
    model <- model_load(model_path, n_gpu_layers = 30, verbosity = 2L)
    cat("✅ Model loaded\\n")
    
    # Create context
    ctx <- context_create(model, n_ctx = 1024, verbosity = 2L)
    cat("✅ Context created\\n")
    
    # Test different parameter combinations
    test_configs <- list(
      list(
        name = "Low_Temp_Short", 
        max_tokens = 20, 
        temperature = 0.1, 
        top_k = 10,
        seed = 42
      ),
      list(
        name = "Med_Temp_Medium", 
        max_tokens = 40, 
        temperature = 0.5,
        top_k = 30,
        seed = 123
      ),
      list(
        name = "High_Temp_Long", 
        max_tokens = 60, 
        temperature = 0.9,
        top_k = 50,
        seed = 456
      )
    )
    
    overall_clean_count <- 0
    overall_test_count <- 0
    
    for (config_idx in seq_along(test_configs)) {
      config <- test_configs[[config_idx]]
      cat(sprintf("\\n📊 Configuration %d: %s\\n", config_idx, config$name))
      cat(sprintf("   Max tokens: %d, Temperature: %.1f, Top-k: %d\\n", 
                  config$max_tokens, config$temperature, config$top_k))
      
      config_clean_count <- 0
      
      for (prompt_idx in seq_along(test_prompts)) {
        prompt <- test_prompts[[prompt_idx]]
        
        cat(sprintf("\\n  🧪 Test %d.%d: '%s'\\n", config_idx, prompt_idx, 
                    substr(prompt, 1, 30), if(nchar(prompt) > 30) "..." else ""))
        
        # RAW tokenization - no chat template, minimal processing
        tokens <- tokenize(model, prompt, add_special = TRUE)
        cat(sprintf("     📝 Tokenized: %d tokens\\n", length(tokens)))
        
        start_time <- Sys.time()
        
        # RAW generation - direct call to bottom-level function
        # NO post-processing, NO cleaning, PURE C++ output
        raw_result <- generate(ctx, tokens,
                              max_tokens = config$max_tokens,
                              temperature = config$temperature,
                              top_k = config$top_k,
                              top_p = 0.9,
                              seed = config$seed,
                              repeat_last_n = 64,
                              penalty_repeat = 1.1)
        
        end_time <- Sys.time()
        generation_time <- as.numeric(end_time - start_time)
        
        cat(sprintf("     ⏱️ Generated in %.2f seconds\\n", generation_time))
        cat(sprintf("     📏 Length: %d characters\\n", nchar(raw_result)))
        cat(sprintf("     📄 RAW OUTPUT: '%s'\\n", 
                    substr(raw_result, 1, 100), if(nchar(raw_result) > 100) "..." else ""))
        
        # Check for EOG contamination in RAW output
        eog_found <- FALSE
        found_patterns <- character(0)
        
        for (pattern in eog_patterns) {
          if (grepl(pattern, raw_result, fixed = TRUE)) {
            eog_found <- TRUE
            found_patterns <- c(found_patterns, pattern)
          }
        }
        
        if (eog_found) {
          cat(sprintf("     ❌ BOTTOM-LEVEL LEAK: %s\\n", paste(found_patterns, collapse = ", ")))
        } else {
          cat("     ✅ BOTTOM-LEVEL CLEAN\\n")
          config_clean_count <- config_clean_count + 1
          overall_clean_count <- overall_clean_count + 1
        }
        
        overall_test_count <- overall_test_count + 1
        
        # Store detailed result
        test_key <- sprintf("%s_prompt_%d", config$name, prompt_idx)
        results[[test_key]] <- list(
          config = config$name,
          prompt = prompt,
          raw_output = raw_result,
          length = nchar(raw_result),
          clean = !eog_found,
          eog_patterns = found_patterns,
          generation_time = generation_time,
          test_type = "bottom_level_raw"
        )
      }
      
      config_success_rate <- 100 * config_clean_count / length(test_prompts)
      cat(sprintf("\\n  📊 %s Summary: %d/%d clean (%.1f%%)\\n", 
                  config$name, config_clean_count, length(test_prompts), config_success_rate))
    }
    
    # Clean up
    rm(model, ctx)
    backend_free()
    
    # Overall summary
    cat("\\n", rep("=", 70), "\\n")
    cat("🔬 BOTTOM-LEVEL MECHANISM TEST RESULTS\\n")
    cat(rep("=", 70), "\\n")
    
    overall_success_rate <- 100 * overall_clean_count / overall_test_count
    cat(sprintf("📈 Overall bottom-level success: %d/%d (%.1f%%)\\n", 
                overall_clean_count, overall_test_count, overall_success_rate))
    
    # Detailed failure analysis
    failed_tests <- results[sapply(results, function(r) !r$clean)]
    if (length(failed_tests) > 0) {
      cat("\\n❌ Bottom-level failures detected:\\n")
      for (test_name in names(failed_tests)) {
        test_result <- failed_tests[[test_name]]
        cat(sprintf("   %s: Found patterns [%s]\\n", 
                    test_name, paste(test_result$eog_patterns, collapse = ", ")))
        cat(sprintf("      Output: '%s'\\n", 
                    substr(test_result$raw_output, 1, 80),
                    if(nchar(test_result$raw_output) > 80) "..." else ""))
      }
    }
    
    # Final conclusion about bottom-level mechanisms
    if (overall_clean_count == overall_test_count) {
      cat("\\n🎉 CONCLUSION: Bottom-level EOG prevention is FULLY EFFECTIVE!\\n")
      cat("✅ The C++ mechanisms successfully prevented ALL EOG leaks.\\n")
      cat("💡 This means the solution is primarily at the core level, not just post-processing.\\n")
    } else if (overall_clean_count == 0) {
      cat("\\n🔴 CONCLUSION: Bottom-level mechanisms are INEFFECTIVE!\\n")
      cat("❌ All raw outputs contained EOG leaks.\\n")
      cat("💡 This suggests the solution relies heavily on post-processing cleanup.\\n")
    } else {
      failure_rate <- 100 * (overall_test_count - overall_clean_count) / overall_test_count
      cat(sprintf("\\n⚠️ CONCLUSION: Bottom-level mechanisms are PARTIALLY EFFECTIVE!\\n"))
      cat(sprintf("📊 %.1f%% failure rate at the core level\\n", failure_rate))
      cat("💡 This suggests a mixed approach with both core and post-processing solutions.\\n")
    }
    
    return(results)
    
  }, error = function(e) {
    cat("❌ Test failed with error:", e$message, "\\n")
    
    # Try to clean up
    tryCatch({ backend_free() }, error = function(e2) {})
    
    return(list(status = "error", error = e$message))
  })
}

# Run the bottom-level test
cat("🚀 Starting bottom-level EOG mechanism test...\\n")
test_results <- test_bottom_level_eog()

cat("\\n🏁 Bottom-level mechanism test completed.\\n")