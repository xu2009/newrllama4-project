# newrllama4

[![R-CMD-check](https://github.com/xu2009/newrllama4-project/workflows/R-CMD-check/badge.svg)](https://github.com/xu2009/newrllama4-project/actions)
[![codecov](https://codecov.io/gh/xu2009/newrllama4-project/branch/main/graph/badge.svg)](https://codecov.io/gh/xu2009/newrllama4-project)
[![CRAN status](https://www.r-pkg.org/badges/version/newrllama4)](https://CRAN.R-project.org/package=newrllama4)

**R Interface to Large Language Models with Runtime Library Loading**

`newrllama4` is an R package that provides access to large language models (LLMs) through the high-performance llama.cpp backend. The package is designed for researchers and data analysts who want to integrate language model capabilities into their R workflows without complex setup requirements.

---

## Quick Start

Get started with large language models in R in three steps:

```r
# Step 1: Install the R package
devtools::install_github("xu2009/newrllama4-project", subdir = "newrllama4")

# Step 2: Download the backend library (first-time setup)
library(newrllama4)
install_newrllama()

# Step 3: Generate text using a language model
response <- quick_llama("Explain the concept of statistical significance")
print(response)
```

---

## Introduction to Large Language Models

Large Language Models (LLMs) are neural networks trained on vast amounts of text data that can understand and generate human language. Key concepts:

- **Large Language Models**: AI systems that process and generate text based on patterns learned from training data
- **GGUF Format**: An efficient binary format for storing and loading language models, optimized for inference
- **Text Generation**: The process of producing coherent text responses based on input prompts

### Research Applications

- **Text Analysis**: Automated content analysis, sentiment analysis, and text classification
- **Data Documentation**: Generate descriptions and summaries of datasets and analytical results  
- **Code Generation**: Produce R code snippets and analysis templates
- **Literature Review**: Summarize research papers and extract key findings
- **Survey Analysis**: Process and categorize open-ended survey responses

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **R Version** | R ≥ 4.0 | R ≥ 4.2 |
| **Memory** | 4GB RAM | 8GB+ RAM |
| **Storage** | 2GB available | 5GB+ available |
| **Network** | Required for downloads | Stable connection |

### Supported Platforms

- **macOS**: Apple Silicon (M1/M2/M3) and Intel processors
- **Windows**: 64-bit systems
- **Linux**: 64-bit distributions (Ubuntu, CentOS, etc.)

### Installation Steps

#### Step 1: Install R Package

```r
# Method 1: Install from GitHub (recommended)
if (!require(devtools)) install.packages("devtools")
devtools::install_github("xu2009/newrllama4-project", subdir = "newrllama4")

# Method 2: From CRAN (when available)
# install.packages("newrllama4")
```

#### Step 2: Download Backend Library

```r
library(newrllama4)

# Automatically detects your system and downloads the appropriate backend (~1MB)
install_newrllama()
```

#### Step 3: Verify Installation

```r
# Check if installation was successful
if (lib_is_installed()) {
  message("Installation successful!")
} else {
  message("Installation failed. See troubleshooting section.")
}
```

### Common Installation Issues

| Issue | Solution |
|-------|----------|
| **Network connection failed** | Check firewall settings or try using a VPN |
| **Insufficient permissions** | Run R as administrator/with elevated privileges |
| **Insufficient disk space** | Free up disk space (minimum 2GB required) |
| **R version too old** | Update to R 4.0 or higher |

---

## Core Functionality

### Level 1: Basic Usage with `quick_llama()`

The simplest way to use the package for text generation:

```r
library(newrllama4)

# Basic text generation
response <- quick_llama("What is machine learning?")
print(response)

# Creative writing with higher temperature
story <- quick_llama("Write a short story about a data scientist", 
                     temperature = 0.9,  # More creative output
                     max_tokens = 200)   # Longer response

# Process multiple questions
questions <- c("Explain regression analysis", 
               "What is clustering?", 
               "Recommend data visualization tools")
answers <- quick_llama(questions)
```

#### Parameter Tuning Guidelines

```r
# Precise answers (suitable for factual questions)
precise <- quick_llama("What is the formula for standard deviation?", 
                       temperature = 0.1,  # Low temperature = more accurate
                       max_tokens = 50)

# Creative responses (suitable for open-ended tasks)
creative <- quick_llama("Design a data analysis project", 
                        temperature = 0.8,  # High temperature = more creative
                        max_tokens = 300)

# Batch processing (for multiple tasks)
prompts <- paste("Analyze the characteristics of dataset", 1:10)
results <- quick_llama(prompts, max_tokens = 100)
```

### Level 2: Intermediate Control - Model Management

When you need more control over the generation process:

```r
# Load a specific model (will download if needed)
model <- model_load("https://huggingface.co/microsoft/DialoGPT-medium/model.gguf")

# Create inference context with custom settings
ctx <- context_create(model, 
                      n_ctx = 4096,     # Longer conversation memory
                      n_threads = 8)    # Use more CPU cores

# Manual text generation
tokens <- tokenize(model, "Please explain deep learning")
result <- generate(ctx, tokens, max_tokens = 150)
text <- detokenize(model, result)
```

### Level 3: Advanced Features

#### Multi-turn Conversations

```r
# Load a chat-optimized model
model <- model_load("path/to/chat_model.gguf")

# Build conversation history
messages <- list(
  list(role = "system", content = "You are a professional data science consultant"),
  list(role = "user", content = "I want to learn R programming"),
  list(role = "assistant", content = "Great choice! R is excellent for statistical analysis..."),
  list(role = "user", content = "Please recommend some beginner books")
)

# Apply chat template formatting
formatted_prompt <- apply_chat_template(model, messages)
response <- quick_llama(formatted_prompt)
```

#### GPU Acceleration Setup

```r
# Enable GPU acceleration (if available)
gpu_model <- model_load("model.gguf", 
                        n_gpu_layers = -1,  # Use all GPU layers
                        use_mlock = TRUE)   # Lock model in memory

# Create context optimized for GPU
ctx_gpu <- context_create(gpu_model, 
                          n_ctx = 8192,     # Larger context window
                          n_threads = 4)    # Fewer CPU threads when using GPU
```

### Level 4: Research Applications

#### Data Analysis Assistant

```r
# Create a data analysis helper function
analyze_data <- function(dataset_description) {
  prompt <- paste("As a data scientist, please analyze the following dataset:", 
                  dataset_description,
                  "Please provide: 1) Data characteristics 2) Analysis recommendations 3) Potential issues")
  
  return(quick_llama(prompt, 
                     temperature = 0.3,  # Maintain analytical rigor
                     max_tokens = 300))
}

# Usage example
result <- analyze_data("E-commerce purchase data with 1000 users, including age, gender, and purchase amount variables")
```

#### R Code Generation

```r
# R code assistant function
generate_r_code <- function(task_description) {
  prompt <- paste("Generate R code to complete the following task:", 
                  task_description,
                  "Return only executable R code with necessary comments.")
  
  return(quick_llama(prompt, temperature = 0.2))
}

# Example usage
code <- generate_r_code("Create a scatter plot showing the relationship between height and weight, including a regression line")
cat(code)
```

---

## Real-world Applications

### 1. Literature Review Assistant

```r
# Literature review helper
literature_review <- function(topic, keywords) {
  prompt <- paste("Create a literature review outline for the following topic:", topic,
                  "Keywords:", paste(keywords, collapse = ", "))
  return(quick_llama(prompt, max_tokens = 400))
}

outline <- literature_review("Machine learning applications in medical diagnosis", 
                           c("deep learning", "medical imaging", "diagnostic accuracy"))
```

### 2. Automated Report Generation

```r
# Generate analysis reports
generate_report <- function(data_summary) {
  prompt <- paste("Generate a professional analysis report based on the following data summary:", 
                  data_summary,
                  "Include key findings, trend analysis, and recommendations.")
  return(quick_llama(prompt, temperature = 0.4, max_tokens = 500))
}
```

### 3. Teaching Assistant

```r
# Statistical concept explainer
explain_concept <- function(concept, level = "beginner") {
  prompt <- paste("Explain the statistical concept", concept, "at a", level, "level.",
                  "Use simple language and practical examples.")
  return(quick_llama(prompt, temperature = 0.3))
}

explanation <- explain_concept("p-value", "beginner")
```

---

## Performance Optimization

### Speed Optimization

```r
# Optimized configuration example
optimized_model <- model_load(
  model_path = "model.gguf",
  n_gpu_layers = -1,        # Full GPU acceleration
  use_mmap = TRUE,          # Memory mapping
  use_mlock = TRUE,         # Lock in memory
  check_memory = TRUE       # Memory checking
)

optimized_ctx <- context_create(
  model = optimized_model,
  n_ctx = 4096,             # Moderate context length
  n_threads = parallel::detectCores() - 1  # Use most CPU cores
)
```

### Memory Management

| Configuration | Memory Usage | Speed | Recommended For |
|---------------|--------------|-------|----------------|
| **CPU Only** | Low | Medium | General tasks, memory constrained |
| **Partial GPU** | Medium | High | Balanced performance and memory |
| **Full GPU** | High | Highest | High-performance requirements |

```r
# Memory-efficient configuration
memory_efficient <- model_load("model.gguf", 
                               n_gpu_layers = 0,     # CPU only
                               use_mmap = TRUE,      # Reduce memory usage
                               use_mlock = FALSE)

# High-performance configuration (requires sufficient memory)
high_performance <- model_load("model.gguf",
                               n_gpu_layers = -1,    # Full GPU
                               use_mlock = TRUE)     # Maximum speed
```

### Parameter Tuning Guidelines

#### Temperature Setting Guide

- **0.1-0.3**: Academic and professional answers (high factual accuracy)
- **0.4-0.6**: Business and daily conversations (balanced)
- **0.7-0.9**: Creative writing and brainstorming
- **1.0+**: Highly creative output (may be incoherent)

#### Max Tokens Selection

- **50-100**: Brief answers and summaries
- **100-300**: Standard explanations and analysis
- **300-500**: Detailed reports and long-form content
- **500+**: Articles and in-depth analysis

---

## Troubleshooting

### Common Errors and Solutions

#### "Backend library not found"

**Cause**: Backend engine not properly installed

**Solution**:
```r
# Reinstall backend
install_newrllama()

# Check installation status
lib_is_installed()

# View library path
get_lib_path()
```

#### "Model loading failed"

**Cause**: Corrupted model file or network issues

**Solution**:
```r
# Force re-download of model
model <- model_load("model_url", force_redownload = TRUE)

# Check disk space
file.info(get_model_cache_dir())
```

#### "Out of memory"

**Cause**: Insufficient system memory

**Solution**:
```r
# Use smaller context size
ctx <- context_create(model, n_ctx = 1024)  # Reduce to 1024

# Close other programs to free memory
# Or use memory-friendly configuration
model <- model_load("model.gguf", n_gpu_layers = 0)
```

### 🐛 调试技巧

```r
# 启用详细日志
options(newrllama.verbose = TRUE)

# 检查系统信息
Sys.info()

# 检查内存使用
memory.size()  # Windows
object.size(model)  # 检查模型大小
```

### Getting Help

1. **Documentation**: `?quick_llama`
2. **Report Issues**: [GitHub Issues](https://github.com/xu2009/newrllama4-project/issues)
3. **Community Discussion**: [GitHub Discussions](https://github.com/xu2009/newrllama4-project/discussions)

---

## 📊 推荐模型指南

### 💻 按用途选择模型

| 用途 | 推荐模型 | 大小 | 特点 |
|------|----------|------|------|
| **快速测试** | Llama-3.2-1B-Instruct | ~1GB | 速度快，基础功能 |
| **日常对话** | Llama-3.2-3B-Instruct | ~2GB | 平衡性能 |
| **专业任务** | Llama-3.1-8B-Instruct | ~5GB | 高质量回答 |
| **代码生成** | CodeLlama-7B-Instruct | ~4GB | 编程专用 |

### 📈 性能vs质量权衡

```r
# 快速原型 - 优先速度
quick_model <- model_load("https://huggingface.co/.../Llama-3.2-1B-Instruct-Q4_K_M.gguf")

# 生产环境 - 优先质量
production_model <- model_load("https://huggingface.co/.../Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# 离线使用 - 本地文件
local_model <- model_load("/path/to/your/model.gguf")
```

### 🌍 多语言支持

```r
# 中文优化模型
chinese_model <- quick_llama("请用中文回答：什么是机器学习？")

# 多语言处理
multilingual <- quick_llama(c(
  "Explain AI in English",
  "Explique l'IA en français", 
  "用中文解释人工智能"
))
```

---

## 🏗️ 架构设计

newrllama4 采用创新的四层架构，平衡了易用性和性能：

```
┌─────────────────────────────────────┐
│        高级R接口 (quick_llama)       │  ← 用户直接使用
├─────────────────────────────────────┤
│     中级API (model_load, generate)  │  ← 高级用户
├─────────────────────────────────────┤
│      R/C++桥接层 (Rcpp接口)        │  ← 数据转换
├─────────────────────────────────────┤
│        C++后端 (llama.cpp)          │  ← 核心计算引擎
└─────────────────────────────────────┘
```

### 为什么这样设计？

- **🎯 易用性**: `quick_llama()` 一行代码即可使用
- **⚡ 性能**: 底层C++引擎确保最高效率  
- **📦 轻量级**: 运行时下载避免大包体积
- **🔧 灵活性**: 支持从简单到复杂的各种用法

---

## 🔬 高级主题

### 自定义模型训练

虽然本包主要用于推理，但您可以使用训练好的GGUF模型：

```r
# 加载您自己训练的模型
custom_model <- model_load("/path/to/your/fine-tuned-model.gguf")

# 或从Hugging Face加载
hf_model <- model_load("hf://your-username/your-model/model.gguf")
```

### Batch Processing Best Practices

```r
# Efficient batch processing
process_batch <- function(prompts, batch_size = 10) {
  results <- list()
  
  for (i in seq(1, length(prompts), batch_size)) {
    batch <- prompts[i:min(i + batch_size - 1, length(prompts))]
    batch_results <- quick_llama(batch)
    results <- c(results, batch_results)
    
    # Progress indicator
    message("Progress: ", min(i + batch_size - 1, length(prompts)), "/", length(prompts))
  }
  
  return(results)
}
```

### 与其他R包集成

```r
# 与ggplot2结合 - AI辅助可视化
library(ggplot2)

plot_suggestion <- quick_llama(
  "基于iris数据集，建议一个有意义的可视化方案，返回ggplot2代码"
)

# 与dplyr结合 - 数据处理建议
library(dplyr)

data_analysis <- quick_llama(
  "为销售数据分析提供dplyr管道操作步骤"
)
```

---

## 🤝 社区与贡献

### 参与贡献

我们欢迎所有形式的贡献！

- 🐛 **报告Bug**: [提交Issue](https://github.com/xu2009/newrllama4-project/issues/new?template=bug_report.md)
- 💡 **功能建议**: [功能请求](https://github.com/xu2009/newrllama4-project/issues/new?template=feature_request.md)
- 📖 **改进文档**: 提交PR改进README和文档
- 💻 **代码贡献**: Fork项目并提交Pull Request

### 开发环境设置

```r
# 开发版本安装
devtools::install_github("xu2009/newrllama4-project", 
                         subdir = "newrllama4", 
                         ref = "develop")

# 运行测试
devtools::test()

# 构建文档
devtools::document()
```

### 行为准则

我们致力于创建一个友好、包容的开源社区。请阅读我们的[行为准则](https://github.com/xu2009/newrllama4-project/blob/main/CODE_OF_CONDUCT.md)。

---

## 📚 相关资源

### 学习资源

- 📖 **R语言**: [R for Data Science](https://r4ds.had.co.nz/)
- 🤖 **AI基础**: [Elements of AI](https://www.elementsofai.com/)
- 🧠 **深度学习**: [Deep Learning with R](https://www.manning.com/books/deep-learning-with-r)

### 相关项目

- 🦙 **llama.cpp**: [原始C++项目](https://github.com/ggerganov/llama.cpp)
- 🤗 **Hugging Face**: [模型库](https://huggingface.co/models)
- 📊 **R语言**: [官方网站](https://www.r-project.org/)

### 技术博客

- [在R中使用大语言模型的最佳实践](https://github.com/xu2009/newrllama4-project/wiki)
- [性能调优指南](https://github.com/xu2009/newrllama4-project/wiki/Performance-Tuning)

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## ⭐ 支持项目

如果这个项目对您有帮助，请考虑：

- ⭐ 给项目加星标
- 🐛 报告问题和建议
- 📢 推荐给朋友和同事
- 💝 [成为赞助者](https://github.com/sponsors/xu2009)

---

## 📞 联系我们

- 📧 **邮箱**: yaoshengleo@example.com
- 🐙 **GitHub**: [@xu2009](https://github.com/xu2009)
- 🐦 **问题反馈**: [GitHub Issues](https://github.com/xu2009/newrllama4-project/issues)

---

<div align="center">

**让AI在R中变得简单而强大** 🚀

[快速开始](#-quick-start---30秒上手) • [查看文档](https://github.com/xu2009/newrllama4-project/wiki) • [报告问题](https://github.com/xu2009/newrllama4-project/issues)

</div>