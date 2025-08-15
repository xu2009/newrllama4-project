# --- FILE: newrllama4/R/zzz.R ---

#' @title R Interface to llama.cpp with Runtime Library Loading
#' @description
#' Provides R bindings to the llama.cpp library for running large language models locally.
#' This package uses an innovative lightweight architecture where the C++ backend library 
#' is downloaded at runtime rather than bundled with the package, enabling zero-configuration
#' AI inference in R with enterprise-grade performance.
#' 
#' @details
#' The newrllama4 package brings state-of-the-art language models to R users through a
#' carefully designed four-layer architecture that combines ease of use with high performance.
#' 
#' ## Quick Start
#' 1. Install the R package: \code{install.packages("newrllama4")}
#' 2. Download backend library: \code{install_newrllama()}
#' 3. Start generating text: \code{quick_llama("Hello, how are you?")}
#' 
#' ## Key Features
#' \itemize{
#'   \item \strong{Zero Configuration}: One-line setup with automatic model downloading
#'   \item \strong{High Performance}: Native C++ inference engine with GPU support
#'   \item \strong{Cross Platform}: Pre-compiled binaries for Windows, macOS, and Linux
#'   \item \strong{Memory Efficient}: Smart caching and memory management
#'   \item \strong{Production Ready}: Robust error handling and comprehensive documentation
#' }
#' 
#' ## Architecture Overview
#' The package uses a layered design:
#' \itemize{
#'   \item \strong{High-Level API}: \code{\link{quick_llama}} for simple text generation
#'   \item \strong{Mid-Level API}: \code{\link{model_load}}, \code{\link{generate}} for detailed control
#'   \item \strong{Low-Level API}: Direct access to tokenization and context management
#'   \item \strong{C++ Backend}: llama.cpp engine with dynamic loading
#' }
#' 
#' ## Main Functions
#' \itemize{
#'   \item \code{\link{install_newrllama}} - Download and install backend library
#'   \item \code{\link{quick_llama}} - High-level text generation (recommended for beginners)
#'   \item \code{\link{model_load}} - Load GGUF models with smart caching
#'   \item \code{\link{context_create}} - Create inference contexts
#'   \item \code{\link{generate}} - Generate text with full parameter control
#'   \item \code{\link{tokenize}} / \code{\link{detokenize}} - Text ↔ Token conversion
#'   \item \code{\link{apply_chat_template}} - Format conversations for chat models
#' }
#' 
#' ## Example Workflows
#' 
#' ### Basic Text Generation
#' \preformatted{
#' # Simple one-liner
#' response <- quick_llama("Explain quantum computing")
#' 
#' # With custom parameters
#' creative_text <- quick_llama("Write a poem about AI", 
#'                              temperature = 0.9, 
#'                              max_tokens = 150)
#' }
#' 
#' ### Advanced Usage with Custom Models
#' \preformatted{
#' # Load your own model
#' model <- model_load("path/to/your/model.gguf")
#' ctx <- context_create(model, n_ctx = 4096)
#' 
#' # Manual tokenization and generation
#' tokens <- tokenize(model, "The future of AI is")
#' output <- generate(ctx, tokens, max_tokens = 100)
#' text <- detokenize(model, output)
#' }
#' 
#' ### Batch Processing
#' \preformatted{
#' # Process multiple prompts efficiently
#' prompts <- c("Summarize AI trends", "Explain machine learning", "What is deep learning?")
#' responses <- quick_llama(prompts)
#' }
#' 
#' ## Supported Model Formats
#' The package works with GGUF format models from various sources:
#' \itemize{
#'   \item Hugging Face Hub (automatic download)
#'   \item Local .gguf files
#'   \item Custom quantized models
#'   \item Ollama-compatible models
#' }
#' 
#' ## Performance Tips
#' \itemize{
#'   \item Use \code{n_gpu_layers = -1} to fully utilize GPU acceleration
#'   \item Set \code{n_threads} to match your CPU cores for optimal performance
#'   \item Use larger \code{n_ctx} values for longer conversations
#'   \item Enable \code{use_mlock} for frequently used models to prevent swapping
#' }
#' 
#' @author yaoshengleo Developer <yaoshengleo@example.com>
#' @references \url{https://github.com/xu2009/newrllama4-project}
#' @keywords package
#' @name newrllama4-package
#' @aliases newrllama4
#' @docType package
"_PACKAGE"

# Environment to store dynamic library information
.pkg_env <- new.env(parent = emptyenv())

.onAttach <- function(libname, pkgname) {
  if (lib_is_installed()) {
    full_lib_path <- get_lib_path()
    
    # Try to load library globally, making symbols available in all DLLs
    tryCatch({
      .pkg_env$lib <- dyn.load(full_lib_path, local = FALSE, now = TRUE)
      
      # NEW STEP: Initialize function pointers at C++ level
      # Get dynamic library handle and initialize API function pointers
      tryCatch({
        # Use the DLL info returned by dyn.load to initialize API
        # We pass the library path, C++ side will reopen it to get handle
        .Call("c_newrllama_api_init", full_lib_path)
        
        packageStartupMessage("newrllama backend library loaded and API initialized successfully")
      }, error = function(e) {
        packageStartupMessage("Warning: Backend library loaded but API initialization failed: ", e$message)
        packageStartupMessage("The library may still work, but some functions might not be available.")
      })
      
    }, error = function(e) {
      packageStartupMessage("Warning: Failed to load backend library: ", e$message)
      packageStartupMessage("Please try reinstalling with: install_newrllama()")
    })
    
  } else {
    # Only show message in interactive sessions to avoid interference during installation
    if (interactive()) {
      packageStartupMessage(
        "Welcome to newrllama4! The backend library is not yet installed.\n",
        "Please run `install_newrllama()` to download and set it up."
      )
    }
  }
}

.onUnload <- function(libpath) {
  # Safely clean up without causing bus errors
  if (!is.null(.pkg_env$lib)) {
    tryCatch({
      # Only attempt cleanup if the package is being properly unloaded
      # Skip API reset to avoid alignment issues
      if (exists("dyn.unload") && is.function(dyn.unload)) {
      dyn.unload(.pkg_env$lib[["path"]])
      }
    }, error = function(e) {
      # Silently handle unload errors
    })
    .pkg_env$lib <- NULL
  }
}

# Helper function: check if library is loaded
.is_backend_loaded <- function() {
  !is.null(.pkg_env$lib)
}

# Helper function: ensure library is loaded
.ensure_backend_loaded <- function() {
  if (!.is_backend_loaded()) {
    if (lib_is_installed()) {
      # Try to reload
      .onAttach(libname = "newrllama4", pkgname = "newrllama4")
    }
    
    if (!.is_backend_loaded()) {
      stop("Backend library is not loaded. Please run install_newrllama() first.", call. = FALSE)
    }
  }
} 