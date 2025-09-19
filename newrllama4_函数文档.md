# newrllama4 R包函数完整参考文档

## 概述
newrllama4是一个高性能的R语言大型语言模型推理包，提供了完整的llama.cpp后端集成。本文档涵盖了包中所有导出函数及其参数的详细说明。

---

## 1. 核心推理函数

### 1.1 `quick_llama()` - 高级文本生成
**简介**: 一站式文本生成函数，自动处理模型加载、聊天模板和生成。

**参数**:
- `prompt` (character): 输入提示文本或字符向量
- `model` (character): 模型路径或URL，默认为Llama-3.2-1B-Instruct
- `n_threads` (integer): CPU线程数，默认自动检测
- `n_gpu_layers` (integer/"auto"): GPU层数，默认"auto"自动检测
- `n_ctx` (integer): 上下文长度，默认2048
- `verbosity` (integer): 输出详细级别(0-3)，默认1L
- `max_tokens` (integer): 最大生成token数，默认100L
- `top_k` (integer): top-k采样参数，默认20L  
- `top_p` (numeric): top-p采样参数，默认0.9
- `temperature` (numeric): 采样温度，默认0.7
- `repeat_last_n` (integer): 重复惩罚考虑的token数，默认64L
- `penalty_repeat` (numeric): 重复惩罚强度，默认1.1
- `min_p` (numeric): 最小概率阈值，默认0.05
- `system_prompt` (character): 系统提示，默认"You are a helpful assistant."
- `auto_format` (logical): 是否自动应用聊天模板，默认TRUE
- `chat_template` (character): 自定义聊天模板，默认NULL
- `stream` (logical): 是否流式输出，默认FALSE
- `seed` (integer): 随机种子，默认1234L

### 1.2 `generate()` - 底层文本生成
**简介**: 使用已创建的上下文进行低级别文本生成。

**参数**:
- `context` (newrllama_context): 推理上下文对象
- `tokens` (integer): 输入token向量
- `max_tokens` (integer): 最大生成token数，默认100L
- `top_k` (integer): top-k采样，默认20L
- `top_p` (numeric): top-p采样，默认0.9
- `temperature` (numeric): 采样温度，默认0.7
- `repeat_last_n` (integer): 重复惩罚窗口，默认64L
- `penalty_repeat` (numeric): 重复惩罚强度，默认1.1
- `seed` (integer): 随机种子，默认1234L

### 1.3 `generate_parallel()` - 并行文本生成
**简介**: 同时处理多个提示进行并行生成。

**参数**:
- `context` (newrllama_context): 推理上下文对象
- `prompts` (character): 提示文本向量
- `max_tokens` (integer): 最大生成token数，默认100L
- `top_k` (integer): top-k采样，默认40L
- `top_p` (numeric): top-p采样，默认0.9
- `temperature` (numeric): 采样温度，默认0.8
- `repeat_last_n` (integer): 重复惩罚窗口，默认64L
- `penalty_repeat` (numeric): 重复惩罚强度，默认1.1
- `seed` (integer): 随机种子，默认-1L(随机)

---

## 2. 模型管理函数

### 2.1 `model_load()` - 模型加载
**简介**: 加载GGUF格式模型，支持自动下载和缓存。

**参数**:
- `model_path` (character): 模型路径或URL(支持https://、hf://等)
- `cache_dir` (character): 自定义缓存目录，默认NULL
- `n_gpu_layers` (integer): GPU层数，默认0L(仅CPU)
- `use_mmap` (logical): 启用内存映射，默认TRUE
- `use_mlock` (logical): 锁定内存防止交换，默认FALSE
- `show_progress` (logical): 显示下载进度，默认TRUE
- `force_redownload` (logical): 强制重新下载，默认FALSE
- `verify_integrity` (logical): 验证文件完整性，默认TRUE
- `check_memory` (logical): 检查内存需求，默认TRUE
- `verbosity` (integer): 输出详细级别，默认1L

### 2.2 `download_model()` - 手动模型下载
**简介**: 手动下载模型到指定位置。

**参数**:
- `model_url` (character): 模型下载URL
- `output_path` (character): 保存路径，默认NULL(使用缓存)
- `show_progress` (logical): 显示下载进度，默认TRUE
- `verify_integrity` (logical): 验证完整性，默认TRUE  
- `max_retries` (integer): 最大重试次数，默认3

### 2.3 `get_model_cache_dir()` - 获取缓存目录
**简介**: 获取模型缓存目录路径。
**参数**: 无

---

## 3. 上下文管理函数

### 3.1 `context_create()` - 创建推理上下文
**简介**: 为已加载模型创建推理上下文。

**参数**:
- `model` (newrllama_model): 已加载的模型对象
- `n_ctx` (integer): 最大上下文长度，默认2048L
- `n_threads` (integer): CPU线程数，默认4L
- `n_seq_max` (integer): 最大并行序列数，默认1L
- `verbosity` (integer): 输出详细级别，默认1L

---

## 4. 文本处理函数

### 4.1 `tokenize()` - 文本分词
**简介**: 将文本转换为token ID序列。

**参数**:
- `model` (newrllama_model): 模型对象
- `text` (character): 待分词文本
- `add_special` (logical): 是否添加特殊token，默认TRUE

### 4.2 `detokenize()` - token解码
**简介**: 将token ID序列转换回文本。

**参数**:
- `model` (newrllama_model): 模型对象
- `tokens` (integer): token ID向量

### 4.3 `tokenize_test()` - 分词测试
**简介**: 测试分词功能的调试函数。

**参数**:
- `model` (newrllama_model): 模型对象

---

## 5. 聊天模板函数

### 5.1 `apply_chat_template()` - 应用聊天模板
**简介**: 使用模型内置聊天模板格式化对话。

**参数**:
- `model` (newrllama_model): 模型对象
- `messages` (list): 消息列表，每个包含role和content字段
- `template` (character): 自定义模板，默认NULL
- `add_assistant` (logical): 是否添加助手提示后缀，默认TRUE

### 5.2 `apply_gemma_chat_template()` - Gemma聊天模板
**简介**: 专为Gemma模型设计的聊天模板格式化。

**参数**:
- `messages` (list): 消息列表
- `add_assistant` (logical): 是否添加助手开始标记，默认TRUE

### 5.3 `smart_chat_template()` - 智能聊天模板
**简介**: 自动检测模型类型并应用合适的聊天模板。

**参数**:
- `model` (newrllama_model): 模型对象
- `messages` (list): 消息列表
- `template` (character): 自定义模板，默认NULL
- `add_assistant` (logical): 是否添加助手前缀，默认TRUE

---

## 6. 后端管理函数

### 6.1 `backend_init()` - 初始化后端
**简介**: 初始化newrllama后端库。
**参数**: 无

### 6.2 `backend_free()` - 释放后端
**简介**: 清理后端资源。
**参数**: 无

### 6.3 `install_newrllama()` - 安装后端库
**简介**: 下载并安装预编译的C++后端库。
**参数**: 无

### 6.4 `lib_is_installed()` - 检查后端安装
**简介**: 检查后端库是否已安装。
**参数**: 无

### 6.5 `get_lib_path()` - 获取库路径
**简介**: 获取已安装后端库的路径。
**参数**: 无

---

## 7. 状态重置函数

### 7.1 `quick_llama_reset()` - 重置quick_llama状态
**简介**: 清除缓存的模型和上下文对象，强制下次调用时重新初始化。
**参数**: 无

---

## 参数设置指南

### 采样参数优化:
- **temperature**: 0.1-0.3(精确)，0.7-0.9(创造性)，>1.0(随机)
- **top_k**: 1-10(保守)，20-50(平衡)，>100(多样)
- **top_p**: 0.8-0.95(常用范围)
- **max_tokens**: 根据需要设置，注意模型上下文限制

### 性能参数优化:
- **n_gpu_layers**: -1(全GPU)，0(全CPU)，正整数(部分GPU)
- **n_threads**: 设置为CPU核心数-1
- **n_ctx**: 更大值支持更长对话但占用更多内存
- **use_mlock**: 频繁使用的模型可启用

### 详细级别设置:
- **verbosity**: 0(全部信息)，1(重要信息)，2(警告和错误)，3(仅错误)

### 模型URL格式:
- `https://example.com/model.gguf` - 直接HTTP下载
- `hf://username/model/file.gguf` - Hugging Face仓库
- `/path/to/local/model.gguf` - 本地文件路径

---

## 典型使用流程

### 简单使用:
```r
# 1. 安装后端(首次)
install_newrllama()

# 2. 生成文本
response <- quick_llama("你好，请介绍一下人工智能")
```

### 高级使用:
```r
# 1. 加载自定义模型
model <- model_load("path/to/model.gguf", n_gpu_layers = -1)

# 2. 创建上下文
ctx <- context_create(model, n_ctx = 4096)

# 3. 分词和生成
tokens <- tokenize(model, "The future of AI")
output <- generate(ctx, tokens, max_tokens = 200, temperature = 0.8)
```

### 批量处理:
```r
# 并行处理多个提示
prompts <- c("解释机器学习", "什么是深度学习", "AI的未来发展")
responses <- quick_llama(prompts, max_tokens = 100)
```

---

本文档涵盖了newrllama4包的所有主要功能。每个函数都经过精心设计，既保证了易用性，又提供了足够的灵活性以满足各种使用场景的需求。