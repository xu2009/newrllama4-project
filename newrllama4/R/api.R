# --- FILE: newrllama4/R/api.R ---

#' Initialize newrllama backend
#' 
#' Initialize the backend library. This should be called once before using other functions.
#' 
#' @export
backend_init <- function() {
  .ensure_backend_loaded()
  invisible(.Call("c_r_backend_init"))
}

#' Free newrllama backend
#' 
#' Clean up backend resources. Usually called automatically.
#' 
#' @export  
backend_free <- function() {
  if (.is_backend_loaded()) {
    invisible(.Call("c_r_backend_free"))
  }
}

#' Load a language model (with smart download)
#'
#' @param model_path Path to the GGUF model file or a URL (supports hf://, ollama://, https://, etc.)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0)
#' @param use_mmap Whether to use memory mapping (default: TRUE)
#' @param use_mlock Whether to use memory locking (default: FALSE)
#' @param show_progress Whether to show download progress for URLs (default: TRUE)
#' @return A model object (external pointer)
#' @export
#' @examples
#' \dontrun{
#' # Load local model
#' model <- model_load("/path/to/model.gguf")
#' 
#' # Auto-download from URL
#' model <- model_load("https://example.com/model.gguf")
#' 
#' # Auto-download from Hugging Face (when implemented)
#' # model <- model_load("hf://microsoft/DialoGPT-medium")
#' }
model_load <- function(model_path, n_gpu_layers = 0L, use_mmap = TRUE, use_mlock = FALSE, show_progress = TRUE) {
  .ensure_backend_loaded()
  
  # Resolve model path (download if needed)
  resolved_path <- .resolve_model_path(model_path, show_progress)
  
  .Call("c_r_model_load", 
        as.character(resolved_path),
        as.integer(n_gpu_layers), 
        as.logical(use_mmap),
        as.logical(use_mlock))
}

#' Create inference context
#'
#' @param model A model object returned by model_load()
#' @param n_ctx Context size (default: 2048)
#' @param n_threads Number of threads (default: 4)  
#' @param n_seq_max Maximum number of sequences (default: 1)
#' @return A context object (external pointer)
#' @export
context_create <- function(model, n_ctx = 2048L, n_threads = 4L, n_seq_max = 1L) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_context_create",
        model,
        as.integer(n_ctx),
        as.integer(n_threads), 
        as.integer(n_seq_max))
}

#' Tokenize text
#'
#' @param model A model object
#' @param text Text to tokenize
#' @param add_special Whether to add special tokens (default: TRUE)
#' @return Integer vector of token IDs
#' @export
tokenize <- function(model, text, add_special = TRUE) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_tokenize",
        model,
        as.character(text),
        as.logical(add_special))
}

#' Detokenize tokens
#'
#' @param model A model object
#' @param tokens Integer vector of token IDs  
#' @return Detokenized text string
#' @export
detokenize <- function(model, tokens) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_detokenize",
        model,
        as.integer(tokens))
}

#' Apply chat template
#'
#' @param model A model object
#' @param messages List of chat messages, each with 'role' and 'content'
#' @param template Optional custom template (default: NULL, use model's template)
#' @param add_assistant Whether to add assistant prompt (default: TRUE)
#' @return Formatted prompt string
#' @export
apply_chat_template <- function(model, messages, template = NULL, add_assistant = TRUE) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_apply_chat_template",
        model,
        template,
        messages,
        as.logical(add_assistant))
}

#' Generate text
#'
#' @param context A context object
#' @param tokens Input tokens or prompt text
#' @param max_tokens Maximum tokens to generate (default: 100)
#' @param top_k Top-k sampling (default: 40)
#' @param top_p Top-p sampling (default: 0.9)
#' @param temperature Sampling temperature (default: 0.8)
#' @param repeat_last_n Repetition penalty last n tokens (default: 64)
#' @param penalty_repeat Repetition penalty strength (default: 1.1)
#' @param seed Random seed (default: -1 for random)
#' @return Generated text
#' @export
generate <- function(context, tokens, max_tokens = 100L, top_k = 40L, top_p = 0.9, 
                     temperature = 0.8, repeat_last_n = 64L, penalty_repeat = 1.1, seed = -1L) {
  .ensure_backend_loaded()
  if (!inherits(context, "newrllama_context")) {
    stop("Expected a newrllama_context object", call. = FALSE)
  }
  
  .Call("c_r_generate",
        context,
        as.integer(tokens),
        as.integer(max_tokens),
        as.integer(top_k),
        as.numeric(top_p),
        as.numeric(temperature),
        as.integer(repeat_last_n),
        as.numeric(penalty_repeat),
        as.integer(seed))
}

#' Generate text in parallel
#'
#' @param context A context object
#' @param prompts Character vector of prompts
#' @param max_tokens Maximum tokens to generate (default: 100)
#' @param top_k Top-k sampling (default: 40)
#' @param top_p Top-p sampling (default: 0.9)
#' @param temperature Sampling temperature (default: 0.8)
#' @param repeat_last_n Repetition penalty last n tokens (default: 64)
#' @param penalty_repeat Repetition penalty strength (default: 1.1)
#' @param seed Random seed (default: -1 for random)
#' @return Character vector of generated texts
#' @export
generate_parallel <- function(context, prompts, max_tokens = 100L, top_k = 40L, top_p = 0.9,
                              temperature = 0.8, repeat_last_n = 64L, penalty_repeat = 1.1, seed = -1L) {
  .ensure_backend_loaded()
  if (!inherits(context, "newrllama_context")) {
    stop("Expected a newrllama_context object", call. = FALSE)
  }
  
  .Call("c_r_generate_parallel",
        context,
        as.character(prompts),
        as.integer(max_tokens),
        as.integer(top_k),
        as.numeric(top_p),
        as.numeric(temperature),
        as.integer(repeat_last_n),
        as.numeric(penalty_repeat),
        as.integer(seed))
}

#' Test tokenize function (debugging)
#'
#' @param model A model object
#' @return Integer vector of tokens for "H"
#' @export
tokenize_test <- function(model) {
  .ensure_backend_loaded()
  if (!inherits(model, "newrllama_model")) {
    stop("Expected a newrllama_model object", call. = FALSE)
  }
  
  .Call("c_r_tokenize_test", model)
}

#' Download a model manually
#'
#' @param model_url URL of the model to download (supports https://, hf://, ollama://, etc.)
#' @param output_path Local path where to save the model (optional, will use cache if not provided)
#' @param show_progress Whether to show download progress (default: TRUE)
#' @return The path where the model was saved
#' @export
#' @examples
#' \dontrun{
#' # Download to specific location
#' download_model("https://example.com/model.gguf", "~/models/my_model.gguf")
#' 
#' # Download to cache (path will be returned)
#' cached_path <- download_model("https://example.com/model.gguf")
#' }
download_model <- function(model_url, output_path = NULL, show_progress = TRUE) {
  .ensure_backend_loaded()
  
  if (is.null(output_path)) {
    output_path <- .get_cache_path(model_url)
  }
  
  # Create output directory if it doesn't exist
  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  message("Downloading model from: ", model_url)
  message("Saving to: ", output_path)
  
  .Call("c_r_download_model", 
        as.character(model_url),
        as.character(output_path),
        as.logical(show_progress))
  
  if (!file.exists(output_path)) {
    stop("Download failed: file not found after download", call. = FALSE)
  }
  
  message("Model downloaded successfully!")
  return(output_path)
}

#' Get the model cache directory
#'
#' @return Path to the directory where models are cached
#' @export
get_model_cache_dir <- function() {
  return(.get_model_cache_dir())
}

# --- Helper Functions for Model Download ---

#' Check if a string represents a URL
#' @param path The path/URL to check
#' @return TRUE if it's a URL, FALSE otherwise
.is_url <- function(path) {
  if (is.null(path) || length(path) == 0 || nchar(path) == 0) {
    return(FALSE)
  }
  
  # Check for common URL protocols
  url_patterns <- c("^https?://", "^hf://", "^huggingface://", "^ollama://", 
                    "^ms://", "^modelscope://", "^github://", "^s3://", "^file://")
  
  for (pattern in url_patterns) {
    if (grepl(pattern, path, ignore.case = TRUE)) {
      return(TRUE)
    }
  }
  
  return(FALSE)
}

#' Get cache directory for models
#' @return Path to the model cache directory
.get_model_cache_dir <- function() {
  cache_dir <- tools::R_user_dir("newrllama4", which = "cache")
  models_dir <- file.path(cache_dir, "models")
  
  if (!dir.exists(models_dir)) {
    dir.create(models_dir, recursive = TRUE)
  }
  
  return(models_dir)
}

#' Generate cache path for a model URL
#' @param model_url The model URL
#' @return Local cache path for the model
.get_cache_path <- function(model_url) {
  cache_dir <- .get_model_cache_dir()
  
  # Extract filename from URL
  # For most URLs, this will be the last part after the final /
  filename <- basename(model_url)
  
  # If no extension or generic name, add .gguf
  if (!grepl("\\.(gguf|bin)$", filename, ignore.case = TRUE)) {
    if (filename == "" || filename == model_url) {
      # Generate a reasonable filename from the URL
      clean_url <- gsub("[^a-zA-Z0-9._-]", "_", model_url)
      filename <- paste0(substr(clean_url, 1, 50), ".gguf")
    } else {
      filename <- paste0(filename, ".gguf")
    }
  }
  
  return(file.path(cache_dir, filename))
}

#' Download a model to cache
#' @param model_url The model URL
#' @param cache_path The local cache path
#' @param show_progress Whether to show download progress
.download_model_to_cache <- function(model_url, cache_path, show_progress = TRUE) {
  .ensure_backend_loaded()
  
  message("Downloading model from: ", model_url)
  message("Saving to: ", cache_path)
  
  .Call("c_r_download_model", 
        as.character(model_url),
        as.character(cache_path),
        as.logical(show_progress))
  
  if (!file.exists(cache_path)) {
    stop("Download failed: file not found after download", call. = FALSE)
  }
  
  message("Model downloaded successfully!")
}

#' Resolve model path (download if needed)
#' @param model_path The model path or URL
#' @param show_progress Whether to show download progress
#' @return The resolved local file path
.resolve_model_path <- function(model_path, show_progress = TRUE) {
  # If it's a local file that exists, return as-is
  if (!.is_url(model_path) && file.exists(model_path)) {
    return(model_path)
  }
  
  # If it's a URL, handle download
  if (.is_url(model_path)) {
    cache_path <- .get_cache_path(model_path)
    
    # If cached version exists, use it
    if (file.exists(cache_path)) {
      message("Using cached model: ", cache_path)
      return(cache_path)
    }
    
    # Download to cache
    .download_model_to_cache(model_path, cache_path, show_progress)
    return(cache_path)
  }
  
  # If it's neither a URL nor an existing file, it's an error
  stop("Model file does not exist and is not a valid URL: ", model_path, call. = FALSE)
} 