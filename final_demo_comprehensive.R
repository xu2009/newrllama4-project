#!/usr/bin/env Rscript
# =============================================================================
# newrllama4 v1.0.41 - Smart Model Download Feature Demo
# =============================================================================

# This script demonstrates the new smart model downloading functionality
# that was added to newrllama4 package. 

cat("=============================================================================\n")
cat("newrllama4 v1.0.41 - Smart Model Download Feature Demo\n")
cat("=============================================================================\n\n")

# Load the package
cat("1. Loading newrllama4 package...\n")
suppressPackageStartupMessages({
  library(newrllama4)
})
cat("✓ Package loaded successfully\n\n")

# Check package version
cat("2. Package information:\n")
cat("Version:", as.character(packageVersion("newrllama4")), "\n")
cat("Cache directory:", get_model_cache_dir(), "\n\n")

# =============================================================================
# Demo 1: URL Detection Capabilities
# =============================================================================
cat("3. URL Detection Capabilities:\n")
cat("The package can now automatically detect various URL formats:\n\n")

test_urls <- c(
  "https://example.com/model.gguf",      # HTTPS
  "http://example.com/model.gguf",       # HTTP  
  "hf://microsoft/DialoGPT-medium",      # Hugging Face
  "ollama://llama2:7b",                  # Ollama
  "file:///absolute/path/model.gguf",    # File protocol
  "/local/path/model.gguf",              # Local path (not URL)
  "model.gguf"                           # Relative path (not URL)
)

for (url in test_urls) {
  is_url <- newrllama4:::.is_url(url)
  status <- if (is_url) "✓ URL" else "✗ Local path"
  cat(sprintf("  %-40s -> %s\n", url, status))
}
cat("\n")

# =============================================================================
# Demo 2: Smart Cache Path Generation
# =============================================================================
cat("4. Smart Cache Path Generation:\n")
cat("URLs are automatically converted to intelligent cache paths:\n\n")

cache_examples <- c(
  "https://example.com/model.gguf",
  "https://raw.githubusercontent.com/user/repo/main/data.csv",
  "hf://microsoft/DialoGPT-medium",
  "ollama://llama2:7b"
)

for (url in cache_examples) {
  cache_path <- newrllama4:::.get_cache_path(url)
  filename <- basename(cache_path)
  cat(sprintf("URL: %s\n", url))
  cat(sprintf("  -> Cache as: %s\n\n", filename))
}

# =============================================================================
# Demo 3: Manual Download Function
# =============================================================================
cat("5. Manual Download Function Demo:\n")
cat("You can manually download models using download_model():\n\n")

# Use a real file from the repository for testing
demo_url <- "https://raw.githubusercontent.com/xu2009/newrllama4-project/main/test_basic_loading.R"
cat("Downloading test file:", demo_url, "\n")

tryCatch({
  downloaded_path <- download_model(demo_url, show_progress = TRUE)
  cat("✓ Download successful!\n")
  cat("  Downloaded to:", downloaded_path, "\n")
  
  if (file.exists(downloaded_path)) {
    size <- file.info(downloaded_path)$size
    cat("  File size:", size, "bytes\n")
  }
}, error = function(e) {
  cat("✗ Download failed:", e$message, "\n")
})
cat("\n")

# =============================================================================
# Demo 4: Smart Model Loading (Simulation)
# =============================================================================
cat("6. Smart Model Loading Integration:\n")
cat("The model_load() function now supports automatic downloading:\n\n")

# This would work with real model URLs, but for demo purposes we show the path resolution
demo_model_url <- "https://raw.githubusercontent.com/xu2009/newrllama4-project/main/test_generation.R"
cat("Example: model_load(\"", demo_model_url, "\")\n\n", sep = "")

cat("What happens behind the scenes:\n")
tryCatch({
  # Show path resolution
  resolved_path <- newrllama4:::.resolve_model_path(demo_model_url, show_progress = FALSE)
  cat("1. URL detected as downloadable resource\n")
  cat("2. Cache path calculated:", basename(resolved_path), "\n")
  if (file.exists(resolved_path)) {
    cat("3. ✓ Using cached version (", file.info(resolved_path)$size, " bytes)\n", sep = "")
  } else {
    cat("3. Would download to cache automatically\n")
  }
  cat("4. Model loading would proceed with local file\n")
}, error = function(e) {
  cat("Error in path resolution:", e$message, "\n")
})
cat("\n")

# =============================================================================
# Demo 5: Cache Management
# =============================================================================
cat("7. Cache Management:\n")
cache_dir <- get_model_cache_dir()
cat("Cache directory:", cache_dir, "\n")

if (dir.exists(cache_dir)) {
  cached_files <- list.files(cache_dir, full.names = FALSE)
  if (length(cached_files) > 0) {
    cat("Cached models:\n")
    for (file in cached_files) {
      full_path <- file.path(cache_dir, file)
      size <- file.info(full_path)$size
      cat(sprintf("  - %-40s (%d bytes)\n", file, size))
    }
  } else {
    cat("No cached models yet\n")
  }
} else {
  cat("Cache directory not yet created\n")
}
cat("\n")

# =============================================================================
# Demo 6: Supported URL Formats
# =============================================================================
cat("8. Supported URL Formats (Current and Future):\n")
cat("✓ HTTPS/HTTP: https://example.com/model.gguf\n")
cat("✓ File protocol: file:///path/to/model.gguf\n")
cat("~ Hugging Face: hf://microsoft/DialoGPT-medium (framework ready)\n")
cat("~ Ollama: ollama://llama2:7b (framework ready)\n")
cat("~ Other protocols can be easily added\n\n")

# =============================================================================
# Summary
# =============================================================================
cat("=============================================================================\n")
cat("FEATURE SUMMARY - newrllama4 v1.0.41\n")
cat("=============================================================================\n")
cat("✓ Smart URL detection for multiple protocols\n")
cat("✓ Automatic model downloading with progress tracking\n")
cat("✓ Intelligent caching system to avoid re-downloads\n")
cat("✓ Seamless integration with existing model_load() function\n")
cat("✓ Manual download function for advanced use cases\n")
cat("✓ Cross-platform compatibility (Windows, macOS, Linux)\n")
cat("✓ Curl-based HTTP downloading with error handling\n")
cat("✓ Cache management utilities\n\n")

cat("USAGE EXAMPLES:\n")
cat("# Automatic download and load:\n")
cat("model <- model_load(\"https://example.com/model.gguf\")\n\n")
cat("# Manual download:\n")
cat("path <- download_model(\"https://example.com/model.gguf\")\n")
cat("model <- model_load(path)\n\n")
cat("# Get cache directory:\n")
cat("cache_dir <- get_model_cache_dir()\n\n")

cat("This feature makes model loading much more convenient by eliminating\n")
cat("the need for manual download steps. Users can now directly provide\n")
cat("URLs to model_load() and the package handles everything automatically!\n\n")

cat("=============================================================================\n")
cat("Demo completed successfully!\n")
cat("=============================================================================\n")