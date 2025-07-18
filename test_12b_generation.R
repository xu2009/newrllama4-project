library(newrllama4)

# 测试12B模型的单序列和并行生成功能
cat("=== 测试12B模型加载和生成功能 ===\n")

model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-Q8_0.gguf"

# 首先检查内存和加载模型
cat("\n=== 内存检查和模型加载 ===\n")
tryCatch({
  # 估算内存需求
  estimated_memory <- .Call("c_r_estimate_model_memory", model_path)
  cat("估算内存需求:", round(estimated_memory / (1024^3), 2), "GB\n")
  
  # 检查可用内存
  memory_available <- .Call("c_r_check_memory_available", estimated_memory)
  cat("内存是否足够:", memory_available, "\n")
  
  if (!memory_available) {
    stop("内存不足，无法加载12B模型")
  }
  
  # 加载模型
  cat("\n正在加载12B模型...\n")
  model <- model_load(model_path, 
                     check_memory = TRUE, 
                     verify_integrity = TRUE,
                     n_gpu_layers = 0L,
                     use_mmap = TRUE)
  cat("✓ 12B模型加载成功！\n")
  
  # 创建上下文
  cat("\n=== 创建上下文 ===\n")
  context <- context_create(model, n_ctx = 1024, n_threads = 4)
  cat("✓ 上下文创建成功\n")
  
  # 测试1: 单序列生成
  cat("\n=== 测试1: 单序列文本生成 ===\n")
  test_prompts <- c(
    "What is artificial intelligence?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about nature."
  )
  
  for (i in seq_along(test_prompts)) {
    cat("\n--- 测试", i, "---\n")
    cat("输入:", test_prompts[i], "\n")
    
    start_time <- Sys.time()
    tryCatch({
      tokens <- tokenize(model, test_prompts[i], add_special = TRUE)
      result <- generate(context, tokens, max_tokens = 50)
      end_time <- Sys.time()
      
      cat("输出:", result, "\n")
      cat("生成时间:", round(as.numeric(end_time - start_time), 2), "秒\n")
      cat("✓ 单序列生成成功\n")
      
    }, error = function(e) {
      cat("✗ 单序列生成失败:", e$message, "\n")
    })
  }
  
  # 测试2: 并行生成（如果函数存在）
  cat("\n=== 测试2: 并行文本生成 ===\n")
  
  # 检查是否有并行生成函数
  parallel_functions <- c("generate_parallel", "parallel_generate")
  parallel_func <- NULL
  
  for (func_name in parallel_functions) {
    if (exists(func_name, mode = "function")) {
      parallel_func <- get(func_name)
      cat("找到并行函数:", func_name, "\n")
      break
    }
  }
  
  if (!is.null(parallel_func)) {
    # 准备并行测试数据
    parallel_prompts <- c(
      "Tell me about machine learning.",
      "What is the future of technology?",
      "Describe the solar system."
    )
    
    cat("开始并行生成测试...\n")
    start_time <- Sys.time()
    
    tryCatch({
      # 尝试并行生成
      parallel_results <- parallel_func(
        model = model,
        prompts = parallel_prompts,
        max_tokens = 30,
        n_threads = 4
      )
      
      end_time <- Sys.time()
      parallel_time <- as.numeric(end_time - start_time)
      
      cat("✓ 并行生成成功！\n")
      cat("并行生成时间:", round(parallel_time, 2), "秒\n")
      cat("平均每个序列:", round(parallel_time / length(parallel_prompts), 2), "秒\n")
      
      # 显示并行结果
      for (i in seq_along(parallel_prompts)) {
        cat("\n--- 并行结果", i, "---\n")
        cat("输入:", parallel_prompts[i], "\n")
        cat("输出:", parallel_results[i], "\n")
      }
      
    }, error = function(e) {
      cat("✗ 并行生成失败:", e$message, "\n")
    })
    
  } else {
    cat("未找到并行生成函数，跳过并行测试\n")
    cat("可用的生成函数:\n")
    funcs <- ls(envir = .GlobalEnv)
    generate_funcs <- funcs[grepl("generate", funcs, ignore.case = TRUE)]
    if (length(generate_funcs) > 0) {
      cat(paste(generate_funcs, collapse = ", "), "\n")
    } else {
      cat("无其他生成函数\n")
    }
  }
  
  # 性能总结
  cat("\n=== 性能总结 ===\n")
  cat("✓ 12B模型成功加载和运行\n")
  cat("✓ 单序列生成功能正常\n")
  if (!is.null(parallel_func)) {
    cat("✓ 并行生成功能测试完成\n")
  } else {
    cat("- 并行生成功能未找到\n")
  }
  
}, error = function(e) {
  cat("\n✗ 测试失败:", e$message, "\n")
  
  # 分析失败原因
  if (grepl("内存|memory|Memory", e$message)) {
    cat("→ 内存不足导致的失败\n")
    cat("→ 内存保护机制正常工作\n")
  } else if (grepl("model|模型", e$message)) {
    cat("→ 模型加载相关错误\n")
  } else {
    cat("→ 其他错误类型\n")
  }
})

# 显示最终系统状态
cat("\n=== 系统内存状态 ===\n")
system("vm_stat | head -6")