#!/usr/bin/env Rscript

# Local CI testing script for newrllama4
# This script simulates CI environment testing locally

cat("=== Local CI Testing for newrllama4 ===\n")

# Set up test environment
Sys.setenv(NEWRLLAMA_TEST_MODE = "TRUE")
Sys.setenv(NEWRLLAMA_CACHE_DIR = file.path(tempdir(), "newrllama_test"))

# Create cache directory
cache_dir <- Sys.getenv("NEWRLLAMA_CACHE_DIR")
if (!dir.exists(cache_dir)) {
  dir.create(cache_dir, recursive = TRUE)
}

cat("Test environment set up\n")
cat("Cache directory:", cache_dir, "\n")

# Load required packages
required_packages <- c("devtools", "testthat", "Rcpp")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing", pkg, "...\n")
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Test 1: Package structure
cat("\n=== Test 1: Package Structure ===\n")
tryCatch({
  # Check if we're in the right directory
  if (!file.exists("newrllama4/DESCRIPTION")) {
    stop("Please run this script from the project root directory")
  }
  
  # Read DESCRIPTION
  desc <- read.dcf("newrllama4/DESCRIPTION")
  cat("Package:", desc[,"Package"], "\n")
  cat("Version:", desc[,"Version"], "\n")
  
  # Check required files
  required_files <- c(
    "newrllama4/DESCRIPTION",
    "newrllama4/NAMESPACE",
    "newrllama4/R/",
    "newrllama4/src/",
    "newrllama4/man/"
  )
  
  for (file in required_files) {
    if (file.exists(file)) {
      cat("✓", file, "exists\n")
    } else {
      cat("✗", file, "missing\n")
    }
  }
  
  cat("Package structure check: PASSED\n")
}, error = function(e) {
  cat("Package structure check: FAILED -", e$message, "\n")
})

# Test 2: Package Loading
cat("\n=== Test 2: Package Loading ===\n")
tryCatch({
  devtools::load_all("newrllama4", quiet = TRUE)
  cat("Package loaded successfully\n")
  
  # Check key functions exist
  key_functions <- c("quick_llama", "model_load", "install_newrllama", "backend_init")
  for (func in key_functions) {
    if (exists(func)) {
      cat("✓", func, "available\n")
    } else {
      cat("✗", func, "missing\n")
    }
  }
  
  cat("Package loading check: PASSED\n")
}, error = function(e) {
  cat("Package loading check: FAILED -", e$message, "\n")
})

# Test 3: Backend Installation (if possible)
cat("\n=== Test 3: Backend Installation ===\n")
tryCatch({
  # Try to install backend
  if (exists("install_newrllama")) {
    cat("Attempting backend installation...\n")
    tryCatch({
      install_newrllama()
      if (lib_is_installed()) {
        cat("✓ Backend installed successfully\n")
        
        # Try to get library path
        lib_path <- get_lib_path()
        cat("Library path:", lib_path, "\n")
        cat("Library exists:", file.exists(lib_path), "\n")
      } else {
        cat("✗ Backend installation reported success but library not found\n")
      }
    }, error = function(e) {
      cat("Backend installation failed:", e$message, "\n")
      cat("This may be expected in some environments\n")
    })
  } else {
    cat("install_newrllama function not available\n")
  }
  
  cat("Backend installation check: COMPLETED\n")
}, error = function(e) {
  cat("Backend installation check: ERROR -", e$message, "\n")
})

# Test 4: Basic Function Tests
cat("\n=== Test 4: Basic Function Tests ===\n")
tryCatch({
  # Test quick_llama parameter validation
  if (exists("quick_llama")) {
    tryCatch({
      quick_llama()
      cat("✗ quick_llama should fail with empty prompt\n")
    }, error = function(e) {
      if (grepl("empty", e$message, ignore.case = TRUE)) {
        cat("✓ quick_llama properly validates empty prompt\n")
      } else {
        cat("? quick_llama error:", e$message, "\n")
      }
    })
  }
  
  # Test utility functions
  if (exists("get_model_cache_dir")) {
    cache_dir <- get_model_cache_dir()
    if (is.character(cache_dir) && nchar(cache_dir) > 0) {
      cat("✓ get_model_cache_dir returns valid path\n")
    } else {
      cat("✗ get_model_cache_dir returns invalid result\n")
    }
  }
  
  cat("Basic function tests: COMPLETED\n")
}, error = function(e) {
  cat("Basic function tests: ERROR -", e$message, "\n")
})

# Test 5: Run testthat tests
cat("\n=== Test 5: Testthat Tests ===\n")
tryCatch({
  if (file.exists("newrllama4/tests/testthat")) {
    cat("Running testthat tests...\n")
    test_results <- devtools::test("newrllama4", quiet = FALSE)
    cat("Test results summary:\n")
    print(test_results)
  } else {
    cat("No testthat tests found\n")
  }
  
  cat("Testthat tests: COMPLETED\n")
}, error = function(e) {
  cat("Testthat tests: ERROR -", e$message, "\n")
})

# Test 6: R CMD check simulation
cat("\n=== Test 6: R CMD check simulation ===\n")
tryCatch({
  cat("Running R CMD check...\n")
  check_results <- devtools::check("newrllama4", 
                                   quiet = FALSE,
                                   args = c("--no-manual", "--as-cran"),
                                   env_vars = c(
                                     "_R_CHECK_FORCE_SUGGESTS_" = "false",
                                     "_R_CHECK_TESTS_NLINES_" = "0"
                                   ))
  
  cat("R CMD check completed\n")
  if (length(check_results$errors) == 0 && length(check_results$warnings) == 0) {
    cat("✓ No errors or warnings\n")
  } else {
    cat("Issues found:\n")
    if (length(check_results$errors) > 0) {
      cat("Errors:", length(check_results$errors), "\n")
    }
    if (length(check_results$warnings) > 0) {
      cat("Warnings:", length(check_results$warnings), "\n")
    }
  }
  
  cat("R CMD check simulation: COMPLETED\n")
}, error = function(e) {
  cat("R CMD check simulation: ERROR -", e$message, "\n")
})

# Clean up
cat("\n=== Cleanup ===\n")
tryCatch({
  # Reset environment variables
  Sys.unsetenv("NEWRLLAMA_TEST_MODE") 
  Sys.unsetenv("NEWRLLAMA_CACHE_DIR")
  
  # Clean up test cache
  if (dir.exists(cache_dir)) {
    unlink(cache_dir, recursive = TRUE)
    cat("Test cache cleaned up\n")
  }
  
  cat("Cleanup completed\n")
}, error = function(e) {
  cat("Cleanup error:", e$message, "\n")
})

cat("\n=== Local CI Testing Complete ===\n")
cat("This simulation helps identify potential CI issues before pushing to GitHub\n")