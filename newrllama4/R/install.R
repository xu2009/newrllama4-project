# --- FILE: newrllama4/R/install.R ---

# Define library version and base URL
.lib_version <- "1.0.34"
.base_url <- "https://github.com/xu2009/newrllama4-project/releases/download/v1.0.34/"

# Get path for local library storage
.lib_path <- function() {
  path <- tools::R_user_dir("newrllama4", which = "data")
  # Include version number in path for future upgrades
  file.path(path, .lib_version) 
}

# Check if library is installed
lib_is_installed <- function() {
  path <- .lib_path()
  # Check if platform-specific library file exists
  sysname <- Sys.info()["sysname"]
  
  if (sysname == "Darwin") {
    # On macOS, look for any dylib file
    lib_files <- list.files(path, pattern = "\\.dylib$", recursive = TRUE)
    return(length(lib_files) > 0)
  } else {
    lib_file <- if (sysname == "Windows") "newrllama.dll" else "libnewrllama.so"
    return(file.exists(file.path(path, lib_file)))
  }
}

# Get full path of installed library
get_lib_path <- function() {
  if (!lib_is_installed()) {
    stop("newrllama backend library is not installed. Please run install_newrllama() first.", call. = FALSE)
  }
  
  path <- .lib_path()
  sysname <- Sys.info()["sysname"]
  
  if (sysname == "Darwin") {
    # On macOS, find the first dylib file
    lib_files <- list.files(path, pattern = "\\.dylib$", recursive = TRUE, full.names = TRUE)
    if (length(lib_files) == 0) {
      stop("Library files not found after installation check passed.", call. = FALSE)
    }
    return(lib_files[1])  # Return the first found dylib file
  } else {
    lib_file <- if (sysname == "Windows") "newrllama.dll" else "libnewrllama.so"
    return(file.path(path, lib_file))
  }
}

# Get platform-specific download URL
.get_download_url <- function() {
  sys <- Sys.info()["sysname"]
  arch <- Sys.info()["machine"]
  
  filename <- NULL
  if (sys == "Darwin") {
    if (arch == "arm64") filename <- "libnewrllama_macos_arm64.zip"
    # else if (arch == "x86_64") filename <- "libnewrllama_macos_x64.zip" # Future expansion
  } else if (sys == "Windows") {
    if (arch == "x86-64") filename <- "newrllama_windows_x64.zip"
  } else if (sys == "Linux") {
    if (arch == "x86_64") filename <- "libnewrllama_linux_x64.zip"
  }
  
  if (is.null(filename)) {
    stop(
      "Your platform (", sys, "/", arch, ") is not currently supported. ",
      "Please open an issue on GitHub for support."
    )
  }
  
  paste0(.base_url, filename)
}

#' Install newrllama Backend Library
#'
#' This function downloads and installs the pre-compiled C++ backend library
#' required for the newrllama4 package to function.
#'
#' @export
install_newrllama <- function() {
  if (lib_is_installed()) {
    message("newrllama backend library is already installed.")
    return(invisible(NULL))
  }
  
  # Get user consent
  if (interactive()) {
    ans <- utils::askYesNo(
      "The newrllama C++ backend library is not installed.
      This will download pre-compiled binaries (~1MB) to your local cache.
      Do you want to proceed?",
      default = TRUE
    )
    if (!ans) {
      stop("Installation cancelled by user.", call. = FALSE)
    }
  }

  lib_dir <- .lib_path()
  if (!dir.exists(lib_dir)) {
    dir.create(lib_dir, recursive = TRUE)
  }
  
  download_url <- .get_download_url()
  dest_file <- file.path(lib_dir, basename(download_url))
  
  message("Downloading from: ", download_url)
  tryCatch({
    utils::download.file(download_url, destfile = dest_file, mode = "wb")
  }, error = function(e) {
    stop("Failed to download backend library. Please check your internet connection.\nError: ", e$message, call. = FALSE)
  })
  
  message("Download complete. Unzipping...")
  utils::unzip(dest_file, exdir = lib_dir)
  unlink(dest_file) # Delete zip file
  
  if (lib_is_installed()) {
    message("newrllama backend library successfully installed to: ", lib_dir)
  } else {
    stop("Installation failed. The library file was not found after unpacking.", call. = FALSE)
  }
  
  invisible(NULL)
} 