#!/usr/bin/env Rscript
# =============================================================================
# Codecov 集成测试脚本
# =============================================================================

cat("🔍 Codecov 集成测试\n")
cat("验证代码覆盖率测试配置\n\n")

# 检查必要的包
required_packages <- c("testthat", "covr")
missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("📦 安装缺失的包:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages)
}

# 检查文件结构
cat("📁 检查文件结构...\n")
files_to_check <- c(
  ".codecov.yml",
  ".github/workflows/test-coverage.yml",
  "newrllama4/tests/testthat.R",
  "newrllama4/tests/testthat/test-install.R",
  "newrllama4/tests/testthat/test-api.R",
  "newrllama4/tests/testthat/test-download.R",
  "README.md"
)

for (file in files_to_check) {
  if (file.exists(file)) {
    cat(sprintf("  ✅ %s\n", file))
  } else {
    cat(sprintf("  ❌ %s - 文件不存在\n", file))
  }
}

# 测试基本的 testthat 功能
cat("\n🧪 测试 testthat 基本功能...\n")
library(testthat)

# 运行一个简单的测试
test_that("basic test works", {
  expect_equal(1 + 1, 2)
  expect_true(TRUE)
  expect_type("hello", "character")
})

cat("  ✅ 基本测试通过\n")

# 检查 newrllama4 包是否可以加载
cat("\n📦 检查 newrllama4 包...\n")
tryCatch({
  library(newrllama4)
  cat("  ✅ newrllama4 包加载成功\n")
  
  # 检查主要函数是否存在
  functions_to_check <- c("backend_init", "backend_free", "model_load", 
                         "context_create", "generate", "generate_parallel")
  
  for (func in functions_to_check) {
    if (exists(func)) {
      cat(sprintf("    ✅ %s 函数存在\n", func))
    } else {
      cat(sprintf("    ❌ %s 函数不存在\n", func))
    }
  }
  
}, error = function(e) {
  cat("  ❌ newrllama4 包加载失败:", e$message, "\n")
  cat("  📝 需要先安装并构建包\n")
})

# 运行包的测试
cat("\n🧪 运行包测试...\n")
if (file.exists("newrllama4/tests/testthat")) {
  tryCatch({
    # 切换到包目录
    original_dir <- getwd()
    setwd("newrllama4")
    
    # 运行测试
    test_results <- testthat::test_dir("tests/testthat", reporter = "summary")
    
    # 回到原目录
    setwd(original_dir)
    
    cat("  ✅ 测试运行完成\n")
    
  }, error = function(e) {
    cat("  ❌ 测试运行失败:", e$message, "\n")
    setwd(original_dir)
  })
} else {
  cat("  ❌ 测试目录不存在\n")
}

# 检查 covr 包
cat("\n📊 检查 covr 包...\n")
if (requireNamespace("covr", quietly = TRUE)) {
  cat("  ✅ covr 包可用\n")
  
  # 测试 covr 基本功能
  tryCatch({
    library(covr)
    cat("  ✅ covr 包加载成功\n")
  }, error = function(e) {
    cat("  ❌ covr 包加载失败:", e$message, "\n")
  })
} else {
  cat("  ❌ covr 包不可用\n")
}

# 提供设置建议
cat("\n💡 Codecov 设置建议:\n")
cat("  1. 确保 GitHub repository 设置正确\n")
cat("  2. 在 GitHub Settings > Secrets 中添加 CODECOV_TOKEN\n")
cat("  3. 在 Codecov.io 上注册并获取 token\n")
cat("  4. 推送代码到 GitHub 触发 Actions\n")
cat("  5. 检查 GitHub Actions 日志确认测试运行\n")

cat("\n📋 接下来的步骤:\n")
cat("  1. git add . && git commit -m \"Add codecov integration\"\n")
cat("  2. git push origin main/master\n")
cat("  3. 访问 https://codecov.io 设置项目\n")
cat("  4. 在 GitHub Settings > Secrets 添加 CODECOV_TOKEN\n")
cat("  5. 等待 GitHub Actions 完成并查看覆盖率报告\n")

cat("\n🎉 Codecov 集成测试完成！\n")