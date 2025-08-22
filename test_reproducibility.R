# ===================================================================
# 可复制性测试 - 验证三个主要函数的种子一致性
# ===================================================================
library(newrllama4)

cat("=== 可复制性测试开始 ===\n\n")

# 初始化模型
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 0)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 64, verbosity = 0)

# ===================================================================
# 1. 单序列生成函数可复制性测试
# ===================================================================
cat("=== 测试1: 单序列生成函数 (generate) ===\n")

test_single_reproducibility <- function() {
  # 固定的测试参数
  test_seed <- 12345L
  test_temperature <- 0.7
  test_max_tokens <- 50
  
  messages <- list(
    list(role = "system", content = "You are a helpful assistant."),
    list(role = "user", content = "Write a short story about a cat in exactly 3 sentences.")
  )
  
  formatted_prompt <- apply_chat_template(model, messages)
  tokens <- tokenize(model, formatted_prompt)
  
  results <- list()
  
  cat("运行参数: seed =", test_seed, ", temperature =", test_temperature, ", max_tokens =", test_max_tokens, "\n")
  
  # 进行3次独立运行
  for(i in 1:3) {
    cat("  运行", i, "...")
    result <- generate(ctx, tokens, 
                      max_tokens = test_max_tokens,
                      temperature = test_temperature,
                      seed = test_seed)
    results[[i]] <- result
    cat(" 完成\n")
  }
  
  # 检查一致性
  all_same <- all(sapply(2:3, function(i) results[[i]] == results[[1]]))
  
  cat("结果一致性:", all_same, "\n")
  
  if(all_same) {
    cat("✅ 单序列生成函数可复制性测试通过\n")
    cat("统一结果:", substr(results[[1]], 1, 100), "...\n")
  } else {
    cat("❌ 单序列生成函数可复制性测试失败\n")
    for(i in 1:3) {
      cat("结果", i, ":", substr(results[[i]], 1, 50), "...\n")
    }
  }
  
  return(list(passed = all_same, results = results))
}

single_test <- test_single_reproducibility()
cat("\n")

# ===================================================================
# 2. 并行序列生成函数可复制性测试  
# ===================================================================
cat("=== 测试2: 并行序列生成函数 (generate_parallel) ===\n")

test_parallel_reproducibility <- function() {
  # 固定的测试参数
  test_seed <- 54321L
  test_temperature <- 0.5
  test_max_tokens <- 30
  
  user_prompts <- c(
    "What is 2+2? Answer with just the number.",
    "Name a primary color.",
    "What is the capital of France?"
  )
  
  # 格式化所有prompts
  formatted_prompts <- sapply(user_prompts, function(prompt) {
    messages <- list(
      list(role = "system", content = "You are a helpful assistant."),
      list(role = "user", content = prompt)
    )
    apply_chat_template(model, messages)
  })
  
  results <- list()
  
  cat("运行参数: seed =", test_seed, ", temperature =", test_temperature, ", max_tokens =", test_max_tokens, "\n")
  cat("测试", length(user_prompts), "个并行prompts\n")
  
  # 进行3次独立运行
  for(i in 1:3) {
    cat("  运行", i, "...")
    result <- generate_parallel(ctx, formatted_prompts,
                               max_tokens = test_max_tokens,
                               temperature = test_temperature,
                               seed = test_seed)
    results[[i]] <- result
    cat(" 完成\n")
  }
  
  # 检查一致性 - 需要逐个对比每个parallel结果
  all_same <- TRUE
  for(j in 1:length(user_prompts)) {
    for(i in 2:3) {
      if(results[[i]][j] != results[[1]][j]) {
        all_same <- FALSE
        break
      }
    }
    if(!all_same) break
  }
  
  cat("结果一致性:", all_same, "\n")
  
  if(all_same) {
    cat("✅ 并行序列生成函数可复制性测试通过\n")
    for(j in 1:length(user_prompts)) {
      cat("  Prompt", j, "统一结果:", substr(results[[1]][j], 1, 40), "...\n")
    }
  } else {
    cat("❌ 并行序列生成函数可复制性测试失败\n")
    for(j in 1:length(user_prompts)) {
      cat("  Prompt", j, ":\n")
      for(i in 1:3) {
        cat("    运行", i, ":", substr(results[[i]][j], 1, 30), "...\n")
      }
    }
  }
  
  return(list(passed = all_same, results = results))
}

parallel_test <- test_parallel_reproducibility()
cat("\n")

# 清理资源准备quick_llama测试
rm(model, ctx)
backend_free()

# ===================================================================
# 3. quick_llama函数可复制性测试
# ===================================================================
cat("=== 测试3: quick_llama函数 ===\n")

test_quick_llama_reproducibility <- function() {
  # 固定的测试参数
  test_seed <- 98765L
  test_temperature <- 0.3
  test_max_tokens <- 40
  test_prompt <- "Explain photosynthesis in simple terms using exactly 2 sentences."
  
  results <- list()
  
  cat("运行参数: seed =", test_seed, ", temperature =", test_temperature, ", max_tokens =", test_max_tokens, "\n")
  cat("测试prompt:", test_prompt, "\n")
  
  # 进行3次独立运行
  for(i in 1:3) {
    cat("  运行", i, "...")
    quick_llama_reset()  # 重置状态确保独立性
    result <- quick_llama(test_prompt,
                         n_gpu_layers = 500L,
                         max_tokens = test_max_tokens,
                         temperature = test_temperature,
                         seed = test_seed,
                         verbosity = 0)
    results[[i]] <- result
    cat(" 完成\n")
  }
  
  # 检查一致性
  all_same <- all(sapply(2:3, function(i) results[[i]] == results[[1]]))
  
  cat("结果一致性:", all_same, "\n")
  
  if(all_same) {
    cat("✅ quick_llama函数可复制性测试通过\n")
    cat("统一结果:", substr(results[[1]], 1, 100), "...\n")
  } else {
    cat("❌ quick_llama函数可复制性测试失败\n")
    for(i in 1:3) {
      cat("结果", i, ":", substr(results[[i]], 1, 50), "...\n")
    }
  }
  
  return(list(passed = all_same, results = results))
}

quick_test <- test_quick_llama_reproducibility()

# ===================================================================
# 4. 额外测试：不同seed应该产生不同结果
# ===================================================================
cat("\n=== 测试4: 验证不同seed产生不同结果 ===\n")

test_different_seeds <- function() {
  cat("测试不同seed是否产生不同结果...\n")
  
  # 使用quick_llama进行快速测试
  prompt <- "Tell me a random fact about space."
  
  quick_llama_reset()
  result1 <- quick_llama(prompt, seed = 111L, max_tokens = 30, temperature = 0.8, verbosity = 0)
  
  quick_llama_reset() 
  result2 <- quick_llama(prompt, seed = 222L, max_tokens = 30, temperature = 0.8, verbosity = 0)
  
  different <- (result1 != result2)
  
  cat("Seed 111结果:", substr(result1, 1, 50), "...\n")
  cat("Seed 222结果:", substr(result2, 1, 50), "...\n")
  cat("结果不同:", different, "\n")
  
  if(different) {
    cat("✅ 不同seed正确产生不同结果\n")
  } else {
    cat("⚠️  不同seed产生了相同结果（可能需要检查）\n")
  }
  
  return(different)
}

different_seeds_test <- test_different_seeds()

# ===================================================================
# 5. 最终汇总报告
# ===================================================================
cat("\n", strrep("=", 60), "\n", sep = "")
cat("=== 可复制性测试总结报告 ===\n")
cat(strrep("=", 60), "\n", sep = "")

tests_passed <- 0
total_tests <- 4

if(single_test$passed) {
  cat("✅ 单序列生成函数 (generate): 可复制\n")
  tests_passed <- tests_passed + 1
} else {
  cat("❌ 单序列生成函数 (generate): 不可复制\n")
}

if(parallel_test$passed) {
  cat("✅ 并行序列生成函数 (generate_parallel): 可复制\n")
  tests_passed <- tests_passed + 1
} else {
  cat("❌ 并行序列生成函数 (generate_parallel): 不可复制\n")
}

if(quick_test$passed) {
  cat("✅ quick_llama函数: 可复制\n")
  tests_passed <- tests_passed + 1
} else {
  cat("❌ quick_llama函数: 不可复制\n")
}

if(different_seeds_test) {
  cat("✅ 不同seed产生不同结果: 正常\n")
  tests_passed <- tests_passed + 1
} else {
  cat("⚠️  不同seed产生不同结果: 需要检查\n")
}

cat("\n总体结果:", tests_passed, "/", total_tests, "测试通过\n")

if(tests_passed == total_tests) {
  cat("🎉 所有可复制性测试全部通过！函数种子控制工作正常。\n")
} else {
  cat("⚠️  部分测试未通过，建议检查种子实现或采样参数。\n")
}

# 清理
backend_free()
cat("\n=== 可复制性测试完成 ===\n")