#!/usr/bin/env Rscript
# 并行生成函数隔离性和污染测试脚本

library(newrllama4)

cat("=== 并行生成函数隔离性测试 ===\n\n")

# 模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  stop("模型文件不存在: ", model_path)
}

cat("加载模型:", model_path, "\n")
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 1)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 1)

# =============================================================================
# 测试1: Batch内序列独立性测试
# =============================================================================
cat("\n=== 测试1: Batch内序列独立性 ===\n")

# 使用明显不同的prompt，检查是否互相影响
prompts_batch1 <- c(
  "Count from 1 to 5: 1, 2, 3, 4, 5. Now continue: 6, 7, 8",
  "Say exactly 'HELLO WORLD' in uppercase.",
  "Complete this math: 2 + 2 ="
)

cat("第一批并行生成:\n")
for(i in seq_along(prompts_batch1)) {
  cat(sprintf("  Prompt %d: %s\n", i, prompts_batch1[i]))
}

results_batch1 <- generate_parallel(ctx, prompts_batch1, max_tokens = 20)

cat("\n第一批结果:\n")
for(i in seq_along(results_batch1)) {
  cat(sprintf("  Result %d: %s\n", i, results_batch1[i]))
}

# 分析独立性
cat("\n独立性分析:\n")
# 检查结果1是否包含数字序列
has_numbers_1 <- grepl("[6-9]|10", results_batch1[1])
cat(sprintf("  结果1包含预期数字序列: %s\n", has_numbers_1))

# 检查结果2是否是大写HELLO WORLD
is_hello_2 <- grepl("HELLO.*WORLD", results_batch1[2], ignore.case = FALSE)
cat(sprintf("  结果2包含HELLO WORLD: %s\n", is_hello_2))

# 检查结果3是否包含数学答案
has_math_3 <- grepl("4", results_batch1[3])
cat(sprintf("  结果3包含数学答案: %s\n", has_math_3))

# 交叉污染检查
cross_contamination <- FALSE
if (grepl("HELLO|WORLD", results_batch1[1]) || grepl("HELLO|WORLD", results_batch1[3])) {
  cross_contamination <- TRUE
  cat("  ⚠️ 发现交叉污染: HELLO WORLD出现在其他结果中\n")
}
if (grepl("[1-5].*[6-9]", results_batch1[2]) || grepl("[1-5].*[6-9]", results_batch1[3])) {
  cross_contamination <- TRUE
  cat("  ⚠️ 发现交叉污染: 数字序列出现在其他结果中\n")
}
if (!cross_contamination) {
  cat("  ✅ 未发现明显的交叉污染\n")
}

# =============================================================================
# 测试2: 上一轮Prompt污染测试
# =============================================================================
cat("\n=== 测试2: 上一轮Prompt污染测试 ===\n")

# 第二批使用完全不同的主题
prompts_batch2 <- c(
  "Name three colors: red, blue, green. What comes next?",
  "Translate 'cat' to Spanish.",
  "What is the capital of France?"
)

cat("第二批并行生成（不同主题）:\n")
for(i in seq_along(prompts_batch2)) {
  cat(sprintf("  Prompt %d: %s\n", i, prompts_batch2[i]))
}

results_batch2 <- generate_parallel(ctx, prompts_batch2, max_tokens = 20)

cat("\n第二批结果:\n")
for(i in seq_along(results_batch2)) {
  cat(sprintf("  Result %d: %s\n", i, results_batch2[i]))
}

# 污染检查
cat("\n上一轮污染分析:\n")
batch2_contamination <- FALSE

# 检查第二批结果是否包含第一批的特征内容
for(i in seq_along(results_batch2)) {
  result <- results_batch2[i]
  
  # 检查是否包含第一批的特征词汇
  if (grepl("HELLO|WORLD", result) || 
      grepl("[1-9].*[1-9].*[1-9]", result) ||  # 数字序列
      grepl("\\+.*=", result)) {  # 数学表达式
    batch2_contamination <- TRUE
    cat(sprintf("  ⚠️ 结果%d可能受到上一轮污染: %s\n", i, result))
  }
}

if (!batch2_contamination) {
  cat("  ✅ 第二批结果未发现上一轮污染\n")
}

# =============================================================================
# 测试3: 相同Prompt一致性测试
# =============================================================================
cat("\n=== 测试3: 相同Prompt一致性测试 ===\n")

# 使用相同prompt进行多次测试，检查结果的一致性
identical_prompt <- "Say 'TEST' and nothing else."
identical_prompts <- rep(identical_prompt, 3)

cat("使用相同prompt进行并行生成:\n")
cat(sprintf("  Prompt (x3): %s\n", identical_prompt))

results_identical <- generate_parallel(ctx, identical_prompts, max_tokens = 10)

cat("\n相同prompt结果:\n")
for(i in seq_along(results_identical)) {
  cat(sprintf("  Result %d: %s\n", i, results_identical[i]))
}

# 一致性分析
cat("\n一致性分析:\n")
all_contain_test <- all(sapply(results_identical, function(x) grepl("TEST", x, ignore.case = TRUE)))
cat(sprintf("  所有结果都包含'TEST': %s\n", all_contain_test))

# 检查结果是否完全相同
all_identical <- length(unique(results_identical)) == 1
cat(sprintf("  所有结果完全相同: %s\n", all_identical))

# =============================================================================
# 测试4: 序列长度影响测试
# =============================================================================
cat("\n=== 测试4: 序列长度影响测试 ===\n")

# 使用不同长度的prompt，检查是否会互相影响
prompts_length_test <- c(
  "Short.",  # 极短
  "This is a medium length prompt that contains more words and should take more processing time to complete the generation.",  # 长
  "Mid."     # 极短
)

cat("不同长度prompt测试:\n")
for(i in seq_along(prompts_length_test)) {
  cat(sprintf("  Prompt %d (%d chars): %s\n", i, nchar(prompts_length_test[i]), 
              if(nchar(prompts_length_test[i]) > 50) paste0(substr(prompts_length_test[i], 1, 47), "...") else prompts_length_test[i]))
}

results_length_test <- generate_parallel(ctx, prompts_length_test, max_tokens = 15)

cat("\n长度测试结果:\n")
for(i in seq_along(results_length_test)) {
  cat(sprintf("  Result %d: %s\n", i, results_length_test[i]))
}

# 检查短prompt是否受长prompt影响
cat("\n长度影响分析:\n")
short_results_clean <- all(nchar(trimws(results_length_test[c(1,3)])) < 100)  # 短prompt结果应该也相对简短
cat(sprintf("  短prompt结果保持简洁: %s\n", short_results_clean))

# =============================================================================
# 测试5: KV Cache隔离测试
# =============================================================================
cat("\n=== 测试5: KV Cache隔离测试 ===\n")

# 使用会建立上下文的prompt
context_prompts <- c(
  "My name is Alice. What is my name?",
  "My name is Bob. What is my name?", 
  "My name is Charlie. What is my name?"
)

cat("上下文隔离测试:\n")
for(i in seq_along(context_prompts)) {
  cat(sprintf("  Prompt %d: %s\n", i, context_prompts[i]))
}

results_context <- generate_parallel(ctx, context_prompts, max_tokens = 20)

cat("\n上下文结果:\n")
for(i in seq_along(results_context)) {
  cat(sprintf("  Result %d: %s\n", i, results_context[i]))
}

# 检查是否正确识别各自的名字
cat("\n上下文隔离分析:\n")
correct_alice <- grepl("Alice", results_context[1], ignore.case = TRUE)
correct_bob <- grepl("Bob", results_context[2], ignore.case = TRUE)
correct_charlie <- grepl("Charlie", results_context[3], ignore.case = TRUE)

cat(sprintf("  Alice上下文正确: %s\n", correct_alice))
cat(sprintf("  Bob上下文正确: %s\n", correct_bob))
cat(sprintf("  Charlie上下文正确: %s\n", correct_charlie))

# 检查是否有名字交叉混淆
context_contamination <- FALSE
if (grepl("Bob|Charlie", results_context[1], ignore.case = TRUE)) {
  context_contamination <- TRUE
  cat("  ⚠️ Alice结果被其他名字污染\n")
}
if (grepl("Alice|Charlie", results_context[2], ignore.case = TRUE)) {
  context_contamination <- TRUE
  cat("  ⚠️ Bob结果被其他名字污染\n")
}
if (grepl("Alice|Bob", results_context[3], ignore.case = TRUE)) {
  context_contamination <- TRUE
  cat("  ⚠️ Charlie结果被其他名字污染\n")
}

if (!context_contamination) {
  cat("  ✅ 上下文隔离良好\n")
}

# =============================================================================
# 综合评估
# =============================================================================
cat("\n=== 综合隔离性评估 ===\n")

tests_passed <- 0
total_tests <- 5

# 统计各项测试结果
cat("测试结果摘要:\n")

if (!cross_contamination) {
  cat("  ✅ Batch内序列独立性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ Batch内序列独立性: 失败\n")
}

if (!batch2_contamination) {
  cat("  ✅ 上一轮Prompt隔离: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ 上一轮Prompt隔离: 失败\n")
}

if (all_contain_test) {
  cat("  ✅ 相同Prompt一致性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ 相同Prompt一致性: 失败\n")
}

if (short_results_clean) {
  cat("  ✅ 序列长度独立性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ 序列长度独立性: 失败\n")
}

if (!context_contamination) {
  cat("  ✅ KV Cache隔离性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ KV Cache隔离性: 失败\n")
}

success_rate <- (tests_passed / total_tests) * 100
cat(sprintf("\n整体隔离性评分: %d/%d (%.0f%%)\n", tests_passed, total_tests, success_rate))

if (success_rate >= 80) {
  cat("🎉 并行生成函数隔离性良好\n")
} else if (success_rate >= 60) {
  cat("⚠️ 并行生成函数隔离性一般，建议优化\n")
} else {
  cat("❌ 并行生成函数隔离性差，需要重大修复\n")
}

# 清理资源
rm(model, ctx)
backend_free()
cat("\n隔离性测试完成。\n")