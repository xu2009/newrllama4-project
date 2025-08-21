#!/usr/bin/env Rscript
# 高级并行生成隔离性测试：序列长度独立性 + Template兼容性

library(newrllama4)

cat("=== 高级并行生成隔离性测试 ===\n\n")

# 模型路径
model_path <- "/Users/yaoshengleo/Desktop/gguf模型/gemma-3-12b-it-q4_0.gguf"

if (!file.exists(model_path)) {
  stop("模型文件不存在: ", model_path)
}

cat("加载模型:", model_path, "\n")
model <- model_load(model_path, n_gpu_layers = 500L, verbosity = 1)
ctx <- context_create(model, n_ctx = 4096, n_seq_max = 512, verbosity = 1)

# =============================================================================
# 测试1: 严格的序列长度独立性测试（包含System Prompt）
# =============================================================================
cat("\n=== 测试1: 严格序列长度独立性（带System Prompt） ===\n")

# 定义不同复杂度的system prompts和user messages
test_cases <- list(
  # 极简case
  simple = list(
    system = "Be brief.",
    user = "Hi",
    expected_pattern = "brief|hello|hi",
    max_tokens = 5
  ),
  
  # 复杂case - 长system prompt + 长user prompt  
  complex = list(
    system = "You are an expert software engineer with 20+ years of experience in distributed systems, microservices architecture, cloud computing, DevOps practices, and full-stack development. You have worked at major tech companies and have deep knowledge of scalable system design, performance optimization, security best practices, and modern development methodologies. Always provide detailed, technically accurate, and well-structured responses.",
    user = "Explain the trade-offs between using a microservices architecture versus a monolithic architecture for a large-scale e-commerce platform, considering factors such as scalability, maintainability, deployment complexity, data consistency, network latency, team structure, and operational overhead. Please provide specific examples and recommendations based on different business contexts.",
    expected_pattern = "microservices|monolithic|architecture",
    max_tokens = 100
  ),
  
  # 另一个极简case
  simple2 = list(
    system = "Be brief.", 
    user = "Bye",
    expected_pattern = "brief|bye|goodbye", 
    max_tokens = 5
  )
)

# 构建带system prompt的messages
messages_list <- list()
expected_patterns <- list()
max_tokens_list <- list()

for(name in names(test_cases)) {
  case <- test_cases[[name]]
  messages <- list(
    list(role = "system", content = case$system),
    list(role = "user", content = case$user)
  )
  messages_list[[name]] <- messages
  expected_patterns[[name]] <- case$expected_pattern
  max_tokens_list[[name]] <- case$max_tokens
}

# 使用apply_chat_template构造prompts
formatted_prompts <- list()
for(name in names(messages_list)) {
  formatted_prompts[[name]] <- apply_chat_template(model, messages_list[[name]])
  cat(sprintf("=== %s Template ===\n", name))
  cat(formatted_prompts[[name]])
  cat("\n")
}

# 并行生成（混合简单和复杂）
mixed_prompts <- c(
  formatted_prompts$simple,
  formatted_prompts$complex, 
  formatted_prompts$simple2
)

mixed_max_tokens <- c(
  test_cases$simple$max_tokens,
  test_cases$complex$max_tokens,
  test_cases$simple2$max_tokens
)

cat("混合长度并行生成:\n")
cat("  简单prompt1 (", nchar(mixed_prompts[1]), " chars, max_tokens=", mixed_max_tokens[1], ")\n")
cat("  复杂prompt (", nchar(mixed_prompts[2]), " chars, max_tokens=", mixed_max_tokens[2], ")\n") 
cat("  简单prompt2 (", nchar(mixed_prompts[3]), " chars, max_tokens=", mixed_max_tokens[3], ")\n")

# 注意：generate_parallel 不支持per-prompt的max_tokens，我们用最大值
overall_max_tokens <- max(mixed_max_tokens)
results_mixed <- generate_parallel(ctx, mixed_prompts, max_tokens = overall_max_tokens)

cat("\n混合长度结果:\n")
for(i in seq_along(results_mixed)) {
  result_length <- nchar(trimws(results_mixed[i]))
  cat(sprintf("  Result %d (%d chars): %s\n", i, result_length, 
              if(result_length > 100) paste0(substr(results_mixed[i], 1, 97), "...") else results_mixed[i]))
}

# 独立性分析
cat("\n序列长度独立性分析:\n")
simple_results_length <- c(nchar(trimws(results_mixed[1])), nchar(trimws(results_mixed[3])))
complex_result_length <- nchar(trimws(results_mixed[2]))

cat(sprintf("  简单prompt结果长度: %d, %d 字符\n", simple_results_length[1], simple_results_length[2]))
cat(sprintf("  复杂prompt结果长度: %d 字符\n", complex_result_length))

# 检查简单prompt是否被复杂prompt"带跑偏"
simple_avg_length <- mean(simple_results_length)
length_ratio <- complex_result_length / simple_avg_length

cat(sprintf("  长度比例 (复杂/简单平均): %.2f\n", length_ratio))

# 独立性判定
length_independence <- TRUE
if (length_ratio < 2.0) {
  cat("  ⚠️ 复杂prompt可能被简单prompt拖累\n")
  length_independence <- FALSE
} else if (simple_avg_length > 200) {
  cat("  ⚠️ 简单prompt可能被复杂prompt影响变长\n")
  length_independence <- FALSE
} else {
  cat("  ✅ 序列长度保持相对独立\n")
}

# =============================================================================
# 测试2: Template兼容性测试
# =============================================================================
cat("\n=== 测试2: Template兼容性测试 ===\n")

# 测试当前apply_chat_template的输出格式
test_message <- list(
  list(role = "system", content = "You are helpful."),
  list(role = "user", content = "Test message")
)

template_output <- apply_chat_template(model, test_message)
cat("当前Template输出:\n")
cat(template_output)
cat("\n")

# 分析template格式
template_analysis <- list(
  is_chatml = grepl("<\\|im_start\\||<\\|im_end\\|", template_output),
  is_gemma = grepl("<start_of_turn>|<end_of_turn>", template_output), 
  is_llama3 = grepl("<\\|start_header_id\\||<\\|end_header_id\\|", template_output),
  is_mistral = grepl("\\[INST\\]|\\[/INST\\]", template_output),
  is_plain = !grepl("<|\\[|\\{\\{", template_output)
)

cat("Template格式分析:\n")
for(format in names(template_analysis)) {
  cat(sprintf("  %s: %s\n", format, template_analysis[[format]]))
}

# 检测主要格式
primary_format <- "unknown"
if(template_analysis$is_chatml) {
  primary_format <- "ChatML"
} else if(template_analysis$is_gemma) {
  primary_format <- "Gemma"  
} else if(template_analysis$is_llama3) {
  primary_format <- "Llama3"
} else if(template_analysis$is_mistral) {
  primary_format <- "Mistral"
} else if(template_analysis$is_plain) {
  primary_format <- "Plain"
}

cat(sprintf("检测到的主要格式: %s\n", primary_format))

# 兼容性评估
compatibility_issues <- list()

if(primary_format == "ChatML" && grepl("gemma", tolower(model_path))) {
  compatibility_issues <- append(compatibility_issues, "Gemma模型使用ChatML格式可能不兼容")
}

if(primary_format == "Gemma" && !grepl("gemma", tolower(model_path))) {
  compatibility_issues <- append(compatibility_issues, "非Gemma模型使用Gemma格式可能不兼容")
}

if(length(compatibility_issues) > 0) {
  cat("⚠️ 发现兼容性问题:\n")
  for(issue in compatibility_issues) {
    cat(sprintf("  - %s\n", issue))
  }
} else {
  cat("✅ 未发现明显的template兼容性问题\n")
}

# =============================================================================
# 测试3: 不同Template格式对并行生成的影响
# =============================================================================
cat("\n=== 测试3: Template格式影响测试 ===\n")

# 手动构造不同格式进行对比测试
formats_to_test <- list(
  # 当前标准格式
  standard = apply_chat_template(model, list(
    list(role = "system", content = "Be concise."),
    list(role = "user", content = "Say 'FORMAT_A'")
  )),
  
  # 手动Gemma格式
  gemma_manual = "<start_of_turn>user\nBe concise.\n\nSay 'FORMAT_B'<end_of_turn>\n<start_of_turn>model\n",
  
  # 简单格式
  simple = "Be concise. Say 'FORMAT_C'"
)

cat("不同格式测试:\n")
for(name in names(formats_to_test)) {
  cat(sprintf("  %s format (%d chars):\n", name, nchar(formats_to_test[[name]])))
  display_text <- formats_to_test[[name]]
  if(nchar(display_text) > 100) {
    display_text <- paste0(substr(display_text, 1, 97), "...")
  }
  cat(sprintf("    %s\n", gsub("\n", "\\n", display_text)))
}

# 并行测试不同格式
format_prompts <- unname(unlist(formats_to_test))
format_results <- generate_parallel(ctx, format_prompts, max_tokens = 20)

cat("\n格式测试结果:\n")
format_names <- names(formats_to_test)
for(i in seq_along(format_results)) {
  cat(sprintf("  %s: %s\n", format_names[i], format_results[i]))
}

# 格式效果分析
cat("\n格式效果分析:\n")
format_scores <- list()
for(i in seq_along(format_results)) {
  result <- format_results[i]
  format_name <- format_names[i]
  
  # 检查是否包含预期的格式标识
  expected_format_id <- paste0("FORMAT_", LETTERS[i])
  contains_expected <- grepl(expected_format_id, result, ignore.case = TRUE)
  
  # 检查是否有停止标记泄漏
  has_leakage <- grepl("<|\\[|\\]>", result)
  
  # 检查响应质量（长度合理性）
  response_length <- nchar(trimws(result))
  length_reasonable <- response_length >= 3 && response_length <= 50
  
  score <- sum(c(contains_expected, !has_leakage, length_reasonable))
  format_scores[[format_name]] <- score
  
  cat(sprintf("  %s: 分数 %d/3 (预期内容:%s, 无泄漏:%s, 长度合理:%s)\n", 
              format_name, score, contains_expected, !has_leakage, length_reasonable))
}

# =============================================================================
# 综合评估
# =============================================================================
cat("\n=== 综合评估 ===\n")

tests_passed <- 0
total_tests <- 3

# 序列长度独立性
if(length_independence) {
  cat("  ✅ 序列长度独立性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ 序列长度独立性: 失败\n")
}

# Template兼容性
template_compatible <- length(compatibility_issues) == 0
if(template_compatible) {
  cat("  ✅ Template兼容性: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ Template兼容性: 失败\n")
}

# 格式效果
best_format_score <- max(unlist(format_scores))
format_effective <- best_format_score >= 2
if(format_effective) {
  cat("  ✅ 格式效果: 通过\n")
  tests_passed <- tests_passed + 1
} else {
  cat("  ❌ 格式效果: 失败\n")
}

success_rate <- (tests_passed / total_tests) * 100
cat(sprintf("\n高级测试评分: %d/%d (%.0f%%)\n", tests_passed, total_tests, success_rate))

# 推荐最佳格式
best_format <- names(format_scores)[which.max(unlist(format_scores))]
cat(sprintf("推荐格式: %s (得分: %d/3)\n", best_format, format_scores[[best_format]]))

if(success_rate >= 80) {
  cat("🎉 高级隔离性测试表现优秀\n")
} else if(success_rate >= 60) {
  cat("⚠️ 高级隔离性测试表现一般\n")
} else {
  cat("❌ 高级隔离性测试需要改进\n")
}

# 清理资源
rm(model, ctx)
backend_free()
cat("\n高级隔离性测试完成。\n")