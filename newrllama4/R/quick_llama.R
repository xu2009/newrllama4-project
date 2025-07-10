# --- FILE: newrllama4/R/quick_llama.R ---

# Package-level globals for caching
.quick_llama_env <- new.env(parent = emptyenv())

#' Quick LLaMA Inference
#'
#' A high-level convenience function that provides one-line LLM inference.
#' Automatically handles model downloading, loading, and text generation.
#'
#' @param prompt Character string or vector of prompts to process
#' @param model Model URL or path (default: Llama-3.2-1B-Instruct Q4_K_M)
#' @param n_threads Number of threads (default: auto-detect)
#' @param n_gpu_layers Number of GPU layers (default: auto-detect)
#' @param n_ctx Context size (default: 2048)
#' @param max_tokens Maximum tokens to generate (default: 100)
#' @param temperature Sampling temperature (default: 0.7)
#' @param top_p Top-p sampling (default: 0.9)
#' @param top_k Top-k sampling (default: 40)
#' @param repeat_penalty Repetition penalty (default: 1.1)
#' @param min_p Minimum probability threshold (default: 0.05)
#' @param stream Whether to stream output (default: auto-detect based on interactive())
#' @param seed Random seed for reproducibility (default: 1234)
#' @param ... Additional parameters passed to generate() or generate_parallel()
#'
#' @return Character string (single prompt) or named list (multiple prompts)
#' @export
#'
#' @examples
#' \dontrun{
#' # Simple usage
#' response <- quick_llama("Hello, how are you?")
#' 
#' # Multiple prompts
#' responses <- quick_llama(c("Summarize AI", "Explain quantum computing"))
#' 
#' # Custom parameters
#' creative_response <- quick_llama("Tell me a story", 
#'                                  temperature = 0.9, 
#'                                  max_tokens = 200)
#' }
quick_llama <- function(prompt,
                        model = .get_default_model(),
                        n_threads = NULL,
                        n_gpu_layers = "auto",
                        n_ctx = 2048L,
                        max_tokens = 100L,
                        temperature = 0.7,
                        top_p = 0.9,
                        top_k = 40L,
                        repeat_penalty = 1.1,
                        min_p = 0.05,
                        stream = NULL,
                        seed = 1234L,
                        ...) {
  
  # Validate inputs
  if (missing(prompt) || is.null(prompt) || length(prompt) == 0) {
    stop("Prompt cannot be empty", call. = FALSE)
  }
  
  # Auto-detect stream mode if not specified
  if (is.null(stream)) {
    stream <- interactive()
  }
  
  # Auto-detect n_threads if not specified
  if (is.null(n_threads)) {
    n_threads <- max(1L, parallel::detectCores() - 1L)
  }
  
  # Auto-detect n_gpu_layers if specified as "auto"
  if (identical(n_gpu_layers, "auto")) {
    n_gpu_layers <- .detect_gpu_layers()
  }
  
  # Ensure backend is ready
  .ensure_quick_llama_ready()
  
  # Load model and context if not cached or if different model
  tryCatch({
    .ensure_model_loaded(model, n_gpu_layers, n_ctx, n_threads)
  }, error = function(e) {
    stop("Failed to load model: ", e$message, call. = FALSE)
  })
  
  # Generate text
  if (length(prompt) == 1) {
    # Single prompt
    .generate_single(prompt, max_tokens, top_k, top_p, temperature, 
                     repeat_penalty, seed, stream, ...)
  } else {
    # Multiple prompts
    .generate_multiple(prompt, max_tokens, top_k, top_p, temperature, 
                       repeat_penalty, seed, stream, ...)
  }
}

#' Reset quick_llama state
#'
#' Clears cached model and context objects, forcing fresh initialization
#' on the next call to quick_llama().
#'
#' @export
quick_llama_reset <- function() {
  if (exists("model", envir = .quick_llama_env)) {
    rm(list = ls(envir = .quick_llama_env), envir = .quick_llama_env)
  }
  message("quick_llama state reset")
  invisible(NULL)
}

# --- Helper Functions ---

#' Get default model URL
#' @return Default model URL
.get_default_model <- function() {
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
}

#' Detect optimal GPU layers
#' @return Integer number of GPU layers
.detect_gpu_layers <- function() {
  # Try to detect GPU support
  # This is a simplified version - in real implementation, you might
  # want to check for specific GPU libraries or capabilities
  
  sysname <- Sys.info()["sysname"]
  
  # Basic heuristic: if on macOS (likely has Metal), use GPU
  # if on Linux/Windows, be more conservative
  if (sysname == "Darwin") {
    # On macOS, assume Metal is available
    return(999L)  # Use all layers on GPU
  } else if (sysname == "Linux") {
    # On Linux, check if NVIDIA GPU tools are available
    # This is a basic check - more sophisticated detection could be added
    nvidia_smi <- Sys.which("nvidia-smi")
    if (nvidia_smi != "") {
      return(999L)
    }
  }
  
  # Default to CPU-only
  return(0L)
}

#' Ensure backend is ready
.ensure_quick_llama_ready <- function() {
  # Check if backend library is installed
  if (!lib_is_installed()) {
    message("Backend library not found. Installing...")
    install_newrllama()
  }
  
  # Initialize backend if not already done
  if (!.is_backend_loaded()) {
    message("Initializing backend...")
    backend_init()
  }
}

#' Ensure model and context are loaded
#' @param model_path Model path or URL
#' @param n_gpu_layers Number of GPU layers
#' @param n_ctx Context size
#' @param n_threads Number of threads
.ensure_model_loaded <- function(model_path, n_gpu_layers, n_ctx, n_threads) {
  # Check if we have a cached model and context for this configuration
  cache_key <- paste0(model_path, "_", n_gpu_layers, "_", n_ctx, "_", n_threads)
  
  if (exists("cache_key", envir = .quick_llama_env) && 
      identical(.quick_llama_env$cache_key, cache_key) &&
      exists("model", envir = .quick_llama_env) &&
      exists("context", envir = .quick_llama_env)) {
    # Model and context already loaded with same configuration
    return()
  }
  
  # Load model
  message("Loading model...")
  model_obj <- model_load(model_path, n_gpu_layers = n_gpu_layers, show_progress = TRUE)
  
  # Create context
  message("Creating context...")
  context_obj <- context_create(model_obj, n_ctx = n_ctx, n_threads = n_threads)
  
  # Cache the objects
  .quick_llama_env$model <- model_obj
  .quick_llama_env$context <- context_obj
  .quick_llama_env$cache_key <- cache_key
  
  message("Model and context ready!")
}

#' Generate text for single prompt
#' @param prompt Single prompt string
#' @param max_tokens Maximum tokens
#' @param top_k Top-k sampling
#' @param top_p Top-p sampling
#' @param temperature Temperature
#' @param repeat_penalty Repetition penalty
#' @param seed Random seed
#' @param stream Whether to stream
#' @param ... Additional parameters
#' @return Generated text string
.generate_single <- function(prompt, max_tokens, top_k, top_p, temperature, 
                             repeat_penalty, seed, stream, ...) {
  
  context <- .quick_llama_env$context
  model <- .quick_llama_env$model
  
  # Tokenize prompt
  tokens <- tokenize(model, prompt, add_special = TRUE)
  
  # Generate
  if (stream) {
    message("Generating (streaming)...")
    # For streaming, we'll use the regular generate function
    # In a full implementation, you might want to add actual streaming support
    result <- generate(context, tokens, 
                      max_tokens = max_tokens,
                      top_k = top_k,
                      top_p = top_p,
                      temperature = temperature,
                      penalty_repeat = repeat_penalty,
                      seed = seed,
                      ...)
    cat(result, "\n")
    return(result)
  } else {
    message("Generating...")
    result <- generate(context, tokens, 
                      max_tokens = max_tokens,
                      top_k = top_k,
                      top_p = top_p,
                      temperature = temperature,
                      penalty_repeat = repeat_penalty,
                      seed = seed,
                      ...)
    return(result)
  }
}

#' Generate text for multiple prompts
#' @param prompts Vector of prompt strings
#' @param max_tokens Maximum tokens
#' @param top_k Top-k sampling
#' @param top_p Top-p sampling
#' @param temperature Temperature
#' @param repeat_penalty Repetition penalty
#' @param seed Random seed
#' @param stream Whether to stream
#' @param ... Additional parameters
#' @return Named list of generated texts
.generate_multiple <- function(prompts, max_tokens, top_k, top_p, temperature, 
                               repeat_penalty, seed, stream, ...) {
  
  context <- .quick_llama_env$context
  
  message("Generating ", length(prompts), " responses...")
  
  if (stream) {
    # For streaming with multiple prompts, generate one by one
    results <- vector("list", length(prompts))
    names(results) <- paste0("prompt_", seq_along(prompts))
    
    for (i in seq_along(prompts)) {
      cat("\n--- Prompt ", i, " ---\n")
      cat("Input: ", prompts[i], "\n")
      cat("Output: ")
      
      result <- .generate_single(prompts[i], max_tokens, top_k, top_p, 
                                temperature, repeat_penalty, seed, 
                                stream = FALSE, ...)
      cat(result, "\n")
      results[[i]] <- result
    }
    
    return(results)
  } else {
    # Use parallel generation for better performance
    results <- generate_parallel(context, prompts,
                                max_tokens = max_tokens,
                                top_k = top_k,
                                top_p = top_p,
                                temperature = temperature,
                                penalty_repeat = repeat_penalty,
                                seed = seed,
                                ...)
    
    # Return as named list
    names(results) <- paste0("prompt_", seq_along(prompts))
    return(results)
  }
}

#' Check if backend is loaded
#' @return TRUE if backend is loaded, FALSE otherwise
.is_backend_loaded <- function() {
  # Simply check if the backend library is installed
  # The actual loading will be handled by ensure_backend_loaded
  tryCatch({
    lib_is_installed()
  }, error = function(e) {
    FALSE
  })
}