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
#' @param cache_dir Custom cache directory for downloaded models (default: NULL, uses system cache)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0)
#' @param use_mmap Whether to use memory mapping (default: TRUE)
#' @param use_mlock Whether to use memory locking (default: FALSE)
#' @param show_progress Whether to show download progress for URLs (default: TRUE)
#' @param force_redownload Force re-download even if cached version exists (default: FALSE)
#' @param verify_integrity Verify file integrity before loading (default: TRUE)
#' @param check_memory Check if sufficient memory is available before loading (default: TRUE)
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
#' # Download to custom cache directory
#' model <- model_load("https://example.com/model.gguf", cache_dir = "~/my_models")
#' 
#' # Force re-download
#' model <- model_load("https://example.com/model.gguf", force_redownload = TRUE)
#' }
model_load <- function(model_path, cache_dir = NULL, n_gpu_layers = 0L, use_mmap = TRUE, 
                       use_mlock = FALSE, show_progress = TRUE, force_redownload = FALSE, 
                       verify_integrity = TRUE, check_memory = TRUE) {
  .ensure_backend_loaded()
  
  # Resolve model path (download if needed)
  resolved_path <- .resolve_model_path(model_path, cache_dir, show_progress, 
                                       force_redownload, verify_integrity)
  
  # Check memory availability before loading
  if (check_memory) {
    .check_model_memory_requirements(resolved_path)
  }
  
  .Call("c_r_model_load_safe", 
        as.character(resolved_path),
        as.integer(n_gpu_layers), 
        as.logical(use_mmap),
        as.logical(use_mlock),
        as.logical(check_memory))
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
#' @param verify_integrity Verify file integrity after download (default: TRUE)
#' @param max_retries Maximum number of download retries (default: 3)
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
download_model <- function(model_url, output_path = NULL, show_progress = TRUE, 
                           verify_integrity = TRUE, max_retries = 3) {
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
  
  # Download with retry mechanism
  .download_with_retry(model_url, output_path, show_progress, max_retries)
  
  # Verify integrity if requested
  if (verify_integrity) {
    if (!.verify_file_integrity(output_path)) {
      file.remove(output_path)
      stop("Downloaded file failed integrity check", call. = FALSE)
    }
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
#' @param cache_dir Custom cache directory (if NULL, uses default)
#' @return Path to the model cache directory
.get_model_cache_dir <- function(cache_dir = NULL) {
  if (!is.null(cache_dir)) {
    if (!dir.exists(cache_dir)) {
      dir.create(cache_dir, recursive = TRUE)
    }
    return(cache_dir)
  }
  
  # Check environment variable
  env_cache <- Sys.getenv("NEWRLLAMA_CACHE_DIR", unset = NA)
  if (!is.na(env_cache) && nchar(env_cache) > 0) {
    if (!dir.exists(env_cache)) {
      dir.create(env_cache, recursive = TRUE)
    }
    return(env_cache)
  }
  
  # Default cache directory
  cache_dir <- tools::R_user_dir("newrllama4", which = "cache")
  models_dir <- file.path(cache_dir, "models")
  
  if (!dir.exists(models_dir)) {
    dir.create(models_dir, recursive = TRUE)
  }
  
  return(models_dir)
}

#' Generate cache path for a model URL
#' @param model_url The model URL
#' @param cache_dir Custom cache directory (optional)
#' @return Local cache path for the model
.get_cache_path <- function(model_url, cache_dir = NULL) {
  cache_dir <- .get_model_cache_dir(cache_dir)
  
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
  # Create output directory if it doesn't exist
  output_dir <- dirname(cache_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  message("Downloading model from: ", model_url)
  message("Saving to: ", cache_path)
  
  # Download with retry mechanism
  .download_with_retry(model_url, cache_path, show_progress)
  
  message("Model downloaded successfully!")
}

#' Resolve model path (download if needed)
#' @param model_path The model path or URL
#' @param cache_dir Custom cache directory (optional)
#' @param show_progress Whether to show download progress
#' @param force_redownload Force re-download even if cached version exists
#' @param verify_integrity Verify file integrity
#' @return The resolved local file path
.resolve_model_path <- function(model_path, cache_dir = NULL, show_progress = TRUE, 
                                force_redownload = FALSE, verify_integrity = TRUE) {
  # If it's a local file that exists, verify and return
  if (!.is_url(model_path) && file.exists(model_path)) {
    if (verify_integrity && !.verify_file_integrity(model_path)) {
      stop("Local model file failed integrity check: ", model_path, call. = FALSE)
    }
    return(model_path)
  }
  
  # If it's a URL, handle download
  if (.is_url(model_path)) {
    cache_path <- .get_cache_path(model_path, cache_dir)
    
    # Check if cached version exists and is valid
    if (file.exists(cache_path) && !force_redownload) {
      if (verify_integrity) {
        if (.verify_file_integrity(cache_path)) {
          message("Using cached model: ", cache_path)
          return(cache_path)
        } else {
          message("Cached model failed integrity check, re-downloading...")
          file.remove(cache_path)
        }
      } else {
        message("Using cached model: ", cache_path)
        return(cache_path)
      }
    }
    
    # Download to cache
    .download_model_to_cache(model_path, cache_path, show_progress)
    
    # Verify downloaded file
    if (verify_integrity && !.verify_file_integrity(cache_path)) {
      file.remove(cache_path)
      stop("Downloaded model failed integrity check", call. = FALSE)
    }
    
    return(cache_path)
  }
  
  # If it's neither a URL nor an existing file, it's an error
  stop("Model file does not exist and is not a valid URL: ", model_path, call. = FALSE)
}

#' Verify file integrity
#' @param file_path Path to the file to verify
#' @param expected_size Expected file size in bytes (optional)
#' @return TRUE if file is valid, FALSE otherwise
.verify_file_integrity <- function(file_path, expected_size = NULL) {
  if (!file.exists(file_path)) {
    return(FALSE)
  }
  
  file_info <- file.info(file_path)
  
  # Check if file is empty or suspiciously small
  if (file_info$size < 1024) {  # Less than 1KB
    return(FALSE)
  }
  
  # Check expected size if provided
  if (!is.null(expected_size) && file_info$size != expected_size) {
    return(FALSE)
  }
  
  # Basic GGUF format check
  if (!.is_valid_gguf_file(file_path)) {
    return(FALSE)
  }
  
  return(TRUE)
}

#' Check if file is a valid GGUF file
#' @param file_path Path to the file to check
#' @return TRUE if valid GGUF file, FALSE otherwise
.is_valid_gguf_file <- function(file_path) {
  tryCatch({
    con <- file(file_path, "rb")
    on.exit(close(con), add = TRUE)
    
    # Read first 4 bytes for GGUF magic number
    magic <- readBin(con, "raw", n = 4)
    
    # GGUF magic number: "GGUF" (0x47474755)
    expected_magic <- as.raw(c(0x47, 0x47, 0x55, 0x46))
    
    return(identical(magic, expected_magic))
  }, error = function(e) {
    return(FALSE)
  })
}

#' Check memory requirements for model loading
#' @param model_path Path to the model file
.check_model_memory_requirements <- function(model_path) {
  .ensure_backend_loaded()
  
  tryCatch({
    # Get estimated memory requirement
    estimated_memory <- .Call("c_r_estimate_model_memory", as.character(model_path))
    
    # Check if sufficient memory is available
    memory_available <- .Call("c_r_check_memory_available", as.numeric(estimated_memory))
    
    if (!memory_available) {
      file_size_mb <- round(file.info(model_path)$size / 1024 / 1024, 1)
      estimated_mb <- round(estimated_memory / 1024 / 1024, 1)
      
      warning("Insufficient memory detected. Model file size: ", file_size_mb, 
              "MB, estimated memory requirement: ", estimated_mb, "MB. ",
              "Loading may cause system instability or crashes.", call. = FALSE)
              
      response <- readline("Do you want to continue anyway? (y/N): ")
      if (tolower(trimws(response)) != "y") {
        stop("Model loading cancelled by user due to insufficient memory", call. = FALSE)
      }
    }
  }, error = function(e) {
    # If memory check fails, issue warning but continue
    warning("Could not check memory requirements: ", e$message, call. = FALSE)
  })
}

#' Download with retry mechanism
#' @param model_url URL to download from
#' @param output_path Local path to save to
#' @param show_progress Whether to show progress
#' @param max_retries Maximum number of retries
.download_with_retry <- function(model_url, output_path, show_progress = TRUE, max_retries = 3) {
  .ensure_backend_loaded()
  
  # Create lock file to prevent concurrent downloads
  lock_file <- paste0(output_path, ".lock")
  
  # Check if another process is downloading
  if (file.exists(lock_file)) {
    message("Another download in progress, waiting...")
    for (i in 1:30) {  # Wait up to 30 seconds
      Sys.sleep(1)
      if (!file.exists(lock_file)) break
    }
    
    if (file.exists(lock_file)) {
      stop("Download timeout: another process seems to be stuck", call. = FALSE)
    }
  }
  
  # Create lock file
  file.create(lock_file)
  on.exit({
    if (file.exists(lock_file)) {
      file.remove(lock_file)
    }
  }, add = TRUE)
  
  last_error <- NULL
  
  for (attempt in 1:max_retries) {
    if (attempt > 1) {
      message("Download attempt ", attempt, " of ", max_retries, "...")
      Sys.sleep(2)  # Brief delay between retries
    }
    
    tryCatch({
      # Try C++ download first
      .Call("c_r_download_model", 
            as.character(model_url),
            as.character(output_path),
            as.logical(show_progress))
      
      # Check if download was successful
      if (file.exists(output_path) && file.info(output_path)$size > 0) {
        return()
      }
      
      stop("Download produced empty file")
      
    }, error = function(e) {
      last_error <<- e
      
      # Clean up partial download
      if (file.exists(output_path)) {
        file.remove(output_path)
      }
      
      # Try R fallback on last attempt
      if (attempt == max_retries) {
        message("C++ download failed, trying R fallback...")
        
        tryCatch({
          utils::download.file(model_url, output_path, mode = "wb", 
                              method = "auto", quiet = !show_progress)
          
          if (file.exists(output_path) && file.info(output_path)$size > 0) {
            return()
          }
          
        }, error = function(e2) {
          last_error <<- e2
        })
      }
    })
  }
  
  # If we get here, all attempts failed
  stop("Download failed after ", max_retries, " attempts. Last error: ", 
       last_error$message, call. = FALSE)
} 