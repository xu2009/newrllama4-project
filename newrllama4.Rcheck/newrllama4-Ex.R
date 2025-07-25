pkgname <- "newrllama4"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
library('newrllama4')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("download_model")
### * download_model

flush(stderr()); flush(stdout())

### Name: download_model
### Title: Download a model manually
### Aliases: download_model

### ** Examples

## Not run: 
##D # Download to specific location
##D download_model("https://example.com/model.gguf", "~/models/my_model.gguf")
##D 
##D # Download to cache (path will be returned)
##D cached_path <- download_model("https://example.com/model.gguf")
## End(Not run)



cleanEx()
nameEx("get_lib_path")
### * get_lib_path

flush(stderr()); flush(stdout())

### Name: get_lib_path
### Title: Get Backend Library Path
### Aliases: get_lib_path

### ** Examples

## Not run: 
##D # Get library path (requires library to be installed)
##D path <- get_lib_path()
##D print(path)
## End(Not run)



cleanEx()
nameEx("install_newrllama")
### * install_newrllama

flush(stderr()); flush(stdout())

### Name: install_newrllama
### Title: Install newrllama Backend Library
### Aliases: install_newrllama

### ** Examples

## Not run: 
##D # Install the backend library
##D install_newrllama()
## End(Not run)



cleanEx()
nameEx("lib_is_installed")
### * lib_is_installed

flush(stderr()); flush(stdout())

### Name: lib_is_installed
### Title: Check if Backend Library is Installed
### Aliases: lib_is_installed

### ** Examples

# Check if backend library is installed
if (lib_is_installed()) {
  message("Backend library is ready")
} else {
  message("Please run install_newrllama() first")
}



cleanEx()
nameEx("model_load")
### * model_load

flush(stderr()); flush(stdout())

### Name: model_load
### Title: Load a language model (with smart download)
### Aliases: model_load

### ** Examples

## Not run: 
##D # Load local model
##D model <- model_load("/path/to/model.gguf")
##D 
##D # Auto-download from URL
##D model <- model_load("https://example.com/model.gguf")
##D 
##D # Download to custom cache directory
##D model <- model_load("https://example.com/model.gguf", cache_dir = "~/my_models")
##D 
##D # Force re-download
##D model <- model_load("https://example.com/model.gguf", force_redownload = TRUE)
## End(Not run)



cleanEx()
nameEx("quick_llama")
### * quick_llama

flush(stderr()); flush(stdout())

### Name: quick_llama
### Title: Quick LLaMA Inference
### Aliases: quick_llama

### ** Examples

## Not run: 
##D # Simple usage
##D response <- quick_llama("Hello, how are you?")
##D 
##D # Multiple prompts
##D responses <- quick_llama(c("Summarize AI", "Explain quantum computing"))
##D 
##D # Custom parameters
##D creative_response <- quick_llama("Tell me a story", 
##D                                  temperature = 0.9, 
##D                                  max_tokens = 200)
## End(Not run)



### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
