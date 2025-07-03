# --- FILE: newrllama4/R/zzz.R ---

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