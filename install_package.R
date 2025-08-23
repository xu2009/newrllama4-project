# Simple installation script for newrllama4 package
# Run this script to install the package locally

# Install devtools if not already installed
if (!require(devtools, quietly = TRUE)) {
  install.packages("devtools")
  library(devtools)
}

# Install the newrllama4 package from local directory
devtools::install("/Users/yaoshengleo/Desktop/newrllama_4_project/newrllama4")

# Load the package and install backend
library(newrllama4)
install_newrllama()

# Verify installation
if (lib_is_installed()) {
  message("✅ Installation successful! Package is ready to use.")
  message("Try: quick_llama('Hello, world!')")
} else {
  message("❌ Installation failed. Please check error messages above.")
}
