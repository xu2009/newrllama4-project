# newrllama4

[![R-CMD-check](https://github.com/xu2009/newrllama4/workflows/R-CMD-check/badge.svg)](https://github.com/xu2009/newrllama4/actions)
[![codecov](https://codecov.io/gh/xu2009/newrllama4/branch/main/graph/badge.svg)](https://codecov.io/gh/xu2009/newrllama4)
[![CRAN status](https://www.r-pkg.org/badges/version/newrllama4)](https://CRAN.R-project.org/package=newrllama4)

R Interface to llama.cpp with Runtime Library Loading and Model Download

## Overview

`newrllama4` provides R bindings to the llama.cpp library for running large language models. This package uses a lightweight architecture where the C++ backend library is downloaded at runtime rather than bundled with the package.

## Features

- **Runtime Library Loading**: Download pre-compiled backend libraries automatically
- **Text Generation**: Generate text using state-of-the-art language models
- **Parallel Processing**: Generate multiple responses simultaneously
- **Model Download**: Automatic model downloading from URLs (https://, hf://, ollama://)
- **Smart Caching**: Intelligent model and library caching
- **GPU Acceleration**: Support for GPU acceleration when available

## Installation

```r
# Install from GitHub
devtools::install_github("xu2009/newrllama4", subdir = "newrllama4")

# After installation, download the backend library
library(newrllama4)
install_newrllama()
```

## Quick Start

```r
library(newrllama4)

# Initialize backend
backend_init()

# Load a model
model <- model_load("path/to/your/model.gguf")

# Create context
context <- context_create(model, n_ctx = 512)

# Generate text
result <- generate(context, "What is machine learning?", max_tokens = 100)
print(result)

# Parallel generation
prompts <- c("What is AI?", "Explain quantum computing.", "What is blockchain?")
results <- generate_parallel(context, prompts, max_tokens = 50)
```

## Architecture

The package uses a four-layer architecture:

1. **R Package Interface**: User-friendly R functions
2. **R/C++ Bridge**: Rcpp interface for R-C++ communication
3. **C-API Abstraction**: Stable C interface over llama.cpp
4. **Core Backend**: llama.cpp engine for model execution

## System Requirements

- R >= 4.0
- C++17 compatible compiler
- libcurl (optional, for model downloading)

## License

MIT + file LICENSE

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

Please report issues at: https://github.com/xu2009/newrllama4/issues