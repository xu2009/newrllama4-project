# newrllama4

[![R-CMD-check](https://github.com/xu2009/newrllama4-project/workflows/R-CMD-check/badge.svg)](https://github.com/xu2009/newrllama4-project/actions)
[![codecov](https://codecov.io/gh/xu2009/newrllama4-project/branch/main/graph/badge.svg)](https://codecov.io/gh/xu2009/newrllama4-project)
[![CRAN status](https://www.r-pkg.org/badges/version/newrllama4)](https://CRAN.R-project.org/package=newrllama4)

## Brief Introduction

The `newrllama4` package lets you run powerful language models directly on your computer using R. You can ask questions, generate text, and analyze data without sending anything to external services. This tutorial will show you how to install the package and start generating text in just a few minutes.

---

## Installation

You need to install both the R package and a small backend library that does the heavy computation.

```r
# Install the R package
install.packages("newrllama4")

# Load the package and install the backend
library(newrllama4)
install_newrllama()
```

That's it! The installation will automatically detect your system and download the right version.

---

## Quick Start

Let's generate your first text response:

```r
# Load the package
library(newrllama4)

# Ask a question and get a response
response <- quick_llama("What is machine learning?")
print(response)
```

The first time you run this, it will download a language model (about 1GB). After that, everything runs locally on your computer.

---

## Running Example

Let's work with a simple example throughout this tutorial. Imagine you have a dataset with customer feedback that you want to analyze:

```r
# Sample customer feedback data
feedback_data <- data.frame(
  customer_id = 1:4,
  feedback = c(
    "The product quality is excellent, very satisfied!",
    "Delivery was slow, but the item was good.",
    "Poor customer service, very disappointed.",
    "Great value for money, will buy again!"
  )
)

print(feedback_data)
```

Now you can analyze each piece of feedback:

```r
# Analyze the sentiment of the first feedback
result <- quick_llama(paste("Analyze the sentiment of this feedback:", 
                           feedback_data$feedback[1]))
print(result)
```

You can also ask for specific information:

```r
# Extract key topics
topics <- quick_llama(paste("What are the main topics in this feedback:", 
                           feedback_data$feedback[2]))
print(topics)
```

---

## For Loop for Running Example

When you have multiple pieces of text to analyze, you can use a simple loop:

```r
# Analyze sentiment for all feedback
sentiment_results <- c()

for(i in 1:nrow(feedback_data)) {
  prompt <- paste("Classify the sentiment as positive, negative, or neutral:", 
                  feedback_data$feedback[i])
  sentiment <- quick_llama(prompt)
  sentiment_results[i] <- sentiment
  
  # Show progress
  cat("Processed feedback", i, "of", nrow(feedback_data), "\n")
}

# Add results to your data
feedback_data$sentiment <- sentiment_results
print(feedback_data)
```

This will process each piece of feedback one by one and show you the progress.

---

## Parallel Generation

When you have many texts to process, you can speed things up by running multiple generations at the same time. Instead of processing one feedback at a time, you can process them all together:

```r
# Create prompts for all feedback at once
all_prompts <- paste("Classify the sentiment as positive, negative, or neutral:", 
                     feedback_data$feedback)

# Process all prompts simultaneously
all_results <- quick_llama(all_prompts)

# Add results to your data
feedback_data$sentiment <- all_results
print(feedback_data)
```

This is much faster than using a loop because the model processes everything in one go.

---

## Customization

You can adjust how the model generates text by changing various settings.

### Temperature / Determinism

Temperature controls how creative or predictable the responses are:

```r
# Very predictable responses (good for factual questions)
factual <- quick_llama("What is the capital of France?", temperature = 0.1)

# More creative responses (good for writing tasks)
creative <- quick_llama("Write a short story about a robot", temperature = 0.9)
```

Use low temperature (0.1-0.3) for factual questions and high temperature (0.7-1.0) for creative writing.

### Model Download

By default, the package downloads a small, fast model. You can use different models:

```r
# Use a specific model (will download automatically)
response <- quick_llama("Explain quantum physics", 
                       model = "https://huggingface.co/microsoft/model.gguf")

# Use a local model file you've already downloaded
response <- quick_llama("Hello", model = "/path/to/your/model.gguf")
```

Larger models give better responses but need more memory and are slower.

### Max Tokens

Tokens are like words or parts of words. You can control how long the response is:

```r
# Short response (about 50 words)
short <- quick_llama("Explain AI", max_tokens = 50)

# Longer response (about 200 words)  
long <- quick_llama("Explain AI", max_tokens = 200)
```

More tokens = longer responses, but also slower generation.

### Context Window (`n_ctx`)

The context window is how much text the model can "remember" at once:

```r
# For short conversations
quick_llama("Hello", n_ctx = 1024)

# For longer conversations with more history
quick_llama("Hello", n_ctx = 4096)
```

Larger context windows let you have longer conversations but use more memory.

### GPU Usage

If you have a graphics card, you can make generation much faster:

```r
# Use your graphics card for faster generation
quick_llama("Tell me a joke", n_gpu_layers = -1)

# Use only your CPU (slower but works everywhere)
quick_llama("Tell me a joke", n_gpu_layers = 0)
```

GPU acceleration can be 5-10 times faster than CPU-only processing.

### Other Parameters

There are many other options you can explore:

```r
# See all available options
?quick_llama

# Example with multiple custom settings
response <- quick_llama(
  "Write a poem about data science",
  temperature = 0.8,      # Creative
  max_tokens = 150,       # Medium length
  top_p = 0.9,           # Sampling method
  seed = 42              # Reproducible results
)
```

Try different combinations to see what works best for your specific use case.

---

## Getting Help

- **Function documentation**: Type `?quick_llama` in R for detailed help
- **Report issues**: [GitHub Issues](https://github.com/xu2009/newrllama4-project/issues)
- **Community discussion**: [GitHub Discussions](https://github.com/xu2009/newrllama4-project/discussions)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.