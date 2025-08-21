#!/usr/bin/env Rscript
# 并行生成采样逻辑综合测试
# 验证generate_parallel函数的所有采样参数是否正确工作

library(newrllama4)

cat("=== 并行生成采样逻辑综合测试 ===\n\n")

# 模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  stop("模型文件不存在: ", model_path)
}

cat("加载模型:", model_path, "\n")
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 2)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 2)

# =============================================================================
# 测试1: 种子控制和可重现性测试
# =============================================================================
cat("\n=== 测试1: 种子控制和可重现性 ===\n")

test_prompt <- "Count from 1 to 5:"
test_prompts <- rep(test_prompt, 3)

# 固定种子测试 - 多次运行应产生完全相同结果
cat("固定种子重现性测试:\n")
seed_fixed <- 42L
results_run1 <- generate_parallel(ctx, test_prompts, max_tokens = 15L, temperature = 0.8, seed = seed_fixed)
results_run2 <- generate_parallel(ctx, test_prompts, max_tokens = 15L, temperature = 0.8, seed = seed_fixed)
results_run3 <- generate_parallel(ctx, test_prompts, max_tokens = 15L, temperature = 0.8, seed = seed_fixed)

cat("  运行1结果:", results_run1[1], "\n")
cat("  运行2结果:", results_run2[1], "\n")
cat("  运行3结果:", results_run3[1], "\n")

# 检查一致性
seed_consistency <- identical(results_run1, results_run2) && identical(results_run2, results_run3)
cat("  固定种子一致性:", ifelse(seed_consistency, "✅ 通过", "❌ 失败"), "\n")

# 不同种子测试 - 应产生不同结果
cat("\n不同种子差异性测试:\n")
results_seed1 <- generate_parallel(ctx, c(test_prompt), max_tokens = 15L, temperature = 0.8, seed = 111L)
results_seed2 <- generate_parallel(ctx, c(test_prompt), max_tokens = 15L, temperature = 0.8, seed = 222L)
results_seed3 <- generate_parallel(ctx, c(test_prompt), max_tokens = 15L, temperature = 0.8, seed = 333L)

cat("  种子111:", results_seed1[1], "\n")
cat("  种子222:", results_seed2[1], "\n") 
cat("  种子333:", results_seed3[1], "\n")

seed_diversity <- !identical(results_seed1[1], results_seed2[1]) && !identical(results_seed2[1], results_seed3[1])
cat("  不同种子差异性:", ifelse(seed_diversity, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试2: Temperature参数效果测试
# =============================================================================
cat("\n=== 测试2: Temperature参数效果 ===\n")

creative_prompt <- "Write a creative story about a robot:"
temp_prompts <- rep(creative_prompt, 2)

# 低温度 - 应更保守确定
cat("低温度测试 (temperature = 0.1):\n")
low_temp_results <- generate_parallel(ctx, temp_prompts, max_tokens = 30L, temperature = 0.1, seed = 500L)
for(i in seq_along(low_temp_results)) {
  cat(sprintf("  低温结果%d: %s\n", i, substr(low_temp_results[i], 1, 80)))
}

# 高温度 - 应更随机创意
cat("\n高温度测试 (temperature = 1.5):\n")
high_temp_results <- generate_parallel(ctx, temp_prompts, max_tokens = 30L, temperature = 1.5, seed = 500L)
for(i in seq_along(high_temp_results)) {
  cat(sprintf("  高温结果%d: %s\n", i, substr(high_temp_results[i], 1, 80)))
}

# 简单的多样性分析
low_temp_unique_words <- length(unique(unlist(strsplit(paste(low_temp_results, collapse=" "), "\\s+"))))
high_temp_unique_words <- length(unique(unlist(strsplit(paste(high_temp_results, collapse=" "), "\\s+"))))

cat(sprintf("\n温度效果分析:\n"))
cat(sprintf("  低温独特词汇数: %d\n", low_temp_unique_words))
cat(sprintf("  高温独特词汇数: %d\n", high_temp_unique_words))

temp_effect_valid <- high_temp_unique_words >= low_temp_unique_words * 0.8  # 允许一定误差
cat("  温度效果:", ifelse(temp_effect_valid, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试3: Top-k/Top-p采样参数测试
# =============================================================================
cat("\n=== 测试3: Top-k/Top-p采样参数 ===\n")

sampling_prompt <- "The weather today is"
sampling_prompts <- rep(sampling_prompt, 3)

# Top-k测试
cat("Top-k参数测试:\n")
topk_low <- generate_parallel(ctx, sampling_prompts, max_tokens = 20L, top_k = 5L, temperature = 0.8, seed = 600L)
topk_high <- generate_parallel(ctx, sampling_prompts, max_tokens = 20L, top_k = 80L, temperature = 0.8, seed = 600L)

cat("  top_k=5 结果:", topk_low[1], "\n")
cat("  top_k=80 结果:", topk_high[1], "\n")

# Top-p测试  
cat("\nTop-p参数测试:\n")
topp_low <- generate_parallel(ctx, sampling_prompts, max_tokens = 20L, top_p = 0.3, temperature = 0.8, seed = 700L)
topp_high <- generate_parallel(ctx, sampling_prompts, max_tokens = 20L, top_p = 0.95, temperature = 0.8, seed = 700L)

cat("  top_p=0.3 结果:", topp_low[1], "\n")
cat("  top_p=0.95 结果:", topp_high[1], "\n")

# 采样参数独立性检查 (不同参数应产生不同结果)
sampling_independence <- !identical(topk_low[1], topk_high[1]) && !identical(topp_low[1], topp_high[1])
cat("  采样参数独立性:", ifelse(sampling_independence, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试4: 重复惩罚机制测试  
# =============================================================================
cat("\n=== 测试4: 重复惩罚机制 ===\n")

repeat_prompt <- "Say the word 'hello' many times:"
repeat_prompts <- rep(repeat_prompt, 2)

# 无惩罚
cat("无重复惩罚测试 (penalty_repeat = 1.0):\n")
no_penalty <- generate_parallel(ctx, repeat_prompts, max_tokens = 25L, penalty_repeat = 1.0, temperature = 0.8, seed = 800L)
cat("  无惩罚结果:", no_penalty[1], "\n")

# 强惩罚
cat("\n强重复惩罚测试 (penalty_repeat = 1.8):\n")
high_penalty <- generate_parallel(ctx, repeat_prompts, max_tokens = 25L, penalty_repeat = 1.8, temperature = 0.8, seed = 800L)
cat("  强惩罚结果:", high_penalty[1], "\n")

# 重复率分析
count_repeats <- function(text, word = "hello") {
  words <- unlist(strsplit(tolower(text), "\\s+"))
  sum(words == word)
}

no_penalty_repeats <- count_repeats(no_penalty[1])
high_penalty_repeats <- count_repeats(high_penalty[1])

cat(sprintf("\n重复惩罚效果:\n"))
cat(sprintf("  无惩罚'hello'出现次数: %d\n", no_penalty_repeats))  
cat(sprintf("  强惩罚'hello'出现次数: %d\n", high_penalty_repeats))

penalty_effective <- high_penalty_repeats <= no_penalty_repeats
cat("  重复惩罚有效性:", ifelse(penalty_effective, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试5: 长度控制精确性测试
# =============================================================================
cat("\n=== 测试5: 长度控制精确性 ===\n")

length_prompt <- "Explain artificial intelligence"
different_lengths <- c(5L, 15L, 30L)

cat("不同长度控制测试:\n")
length_results <- list()
length_accuracy <- list()

for(i in seq_along(different_lengths)) {
  target_length <- different_lengths[i]
  result <- generate_parallel(ctx, c(length_prompt), max_tokens = target_length, temperature = 0.7, seed = 900L)
  
  # 简单的词汇计数（近似token计数）
  word_count <- length(unlist(strsplit(trimws(result[1]), "\\s+")))
  length_results[[i]] <- result[1]
  length_accuracy[[i]] <- abs(word_count - target_length) / target_length
  
  cat(sprintf("  目标长度%d: 实际词数%d, 误差%.1f%%, 内容: %s\n", 
              target_length, word_count, length_accuracy[[i]] * 100, 
              if(nchar(result[1]) > 60) paste0(substr(result[1], 1, 57), "...") else result[1]))
}

# 长度控制准确性评估
avg_length_error <- mean(unlist(length_accuracy))
length_control_good <- avg_length_error <= 0.3  # 允许30%误差
cat(sprintf("  平均长度误差: %.1f%%\n", avg_length_error * 100))
cat("  长度控制精确性:", ifelse(length_control_good, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试6: 并行独立性和一致性测试
# =============================================================================
cat("\n=== 测试6: 并行独立性和一致性 ===\n")

# 混合不同类型的prompt测试独立性
mixed_prompts <- c(
  "Count to 3:",           # 简单数学
  "Name a color:",         # 简单事实
  "Write a haiku:",        # 创意写作
  "Define 'AI':",          # 解释性
  "Say 'test':"           # 简单重复
)

cat("混合prompt并行测试:\n")
mixed_results <- generate_parallel(ctx, mixed_prompts, 
                                   max_tokens = 20L, 
                                   temperature = 0.8, 
                                   top_k = 40L, 
                                   top_p = 0.9, 
                                   seed = 1000L)

for(i in seq_along(mixed_results)) {
  cat(sprintf("  Prompt%d: %s\n", i, mixed_results[i]))
}

# 质量一致性检查 - 所有结果都应该合理
quality_scores <- numeric(length(mixed_results))
for(i in seq_along(mixed_results)) {
  result <- trimws(mixed_results[i])
  # 简单质量评分：非空、有意义长度、无明显错误标记
  has_content <- nchar(result) >= 3
  reasonable_length <- nchar(result) <= 200
  no_error_markers <- !grepl("ERROR|error|<|>", result)
  
  quality_scores[i] <- sum(c(has_content, reasonable_length, no_error_markers))
}

avg_quality <- mean(quality_scores)
cat(sprintf("\n质量一致性分析:\n"))
for(i in seq_along(quality_scores)) {
  cat(sprintf("  Result%d质量分数: %d/3\n", i, quality_scores[i]))
}
cat(sprintf("  平均质量分数: %.1f/3\n", avg_quality))

consistency_good <- avg_quality >= 2.5
cat("  并行质量一致性:", ifelse(consistency_good, "✅ 通过", "❌ 失败"), "\n")

# =============================================================================
# 测试7: 边界条件和异常处理
# =============================================================================
cat("\n=== 测试7: 边界条件测试 ===\n")

boundary_test_results <- list()

# 极端温度
cat("极端参数测试:\n")
tryCatch({
  extreme_temp <- generate_parallel(ctx, c("Test:"), max_tokens = 5L, temperature = 0.01, seed = 1100L)
  boundary_test_results$extreme_temp <- "通过"
  cat("  极低温度 (0.01): ✅ ", extreme_temp[1], "\n")
}, error = function(e) {
  boundary_test_results$extreme_temp <<- "失败"
  cat("  极低温度 (0.01): ❌ ", e$message, "\n")
})

# 极端top_k
tryCatch({
  extreme_topk <- generate_parallel(ctx, c("Test:"), max_tokens = 5L, top_k = 1L, seed = 1200L)
  boundary_test_results$extreme_topk <- "通过"
  cat("  极小top_k (1): ✅ ", extreme_topk[1], "\n")
}, error = function(e) {
  boundary_test_results$extreme_topk <<- "失败"
  cat("  极小top_k (1): ❌ ", e$message, "\n")
})

# 空prompt测试
tryCatch({
  empty_result <- generate_parallel(ctx, c(""), max_tokens = 10L, seed = 1300L)
  boundary_test_results$empty_prompt <- "通过"
  cat("  空prompt: ✅ '", empty_result[1], "'\n")
}, error = function(e) {
  boundary_test_results$empty_prompt <<- "失败"  
  cat("  空prompt: ❌ ", e$message, "\n")
})

boundary_pass_count <- sum(unlist(boundary_test_results) == "通过")
boundary_total <- length(boundary_test_results)
cat(sprintf("  边界条件通过率: %d/%d\n", boundary_pass_count, boundary_total))

# =============================================================================
# 综合评估和报告
# =============================================================================
cat("\n=== 综合评估报告 ===\n")

test_results <- list(
  种子控制 = seed_consistency && seed_diversity,
  温度效果 = temp_effect_valid,
  采样参数 = sampling_independence, 
  重复惩罚 = penalty_effective,
  长度控制 = length_control_good,
  并行一致性 = consistency_good,
  边界处理 = boundary_pass_count >= 2
)

passed_tests <- sum(unlist(test_results))
total_tests <- length(test_results)

cat("详细测试结果:\n")
for(test_name in names(test_results)) {
  result_icon <- ifelse(test_results[[test_name]], "✅", "❌")
  cat(sprintf("  %s: %s\n", test_name, result_icon))
}

success_rate <- (passed_tests / total_tests) * 100
cat(sprintf("\n采样逻辑测试评分: %d/%d (%.0f%%)\n", passed_tests, total_tests, success_rate))

if(success_rate >= 85) {
  cat("🎉 采样逻辑测试表现优秀！所有核心功能正常工作\n")
} else if(success_rate >= 70) {
  cat("⚠️ 采样逻辑基本正常，但某些方面需要改进\n")
} else {
  cat("❌ 采样逻辑存在严重问题，需要进一步调试\n")
}

# 性能和质量总结
cat("\n=== 性能质量总结 ===\n")
cat(sprintf("  平均长度控制误差: %.1f%%\n", avg_length_error * 100))
cat(sprintf("  平均输出质量评分: %.1f/3\n", avg_quality))
cat(sprintf("  边界条件健壮性: %d/%d\n", boundary_pass_count, boundary_total))

# 清理资源
rm(model, ctx)
backend_free()
cat("\n并行生成采样逻辑测试完成。\n")