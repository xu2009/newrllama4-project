#!/usr/bin/env Rscript

# 检查newrllama4包中的可用函数
library(newrllama4)

cat("=== 检查newrllama4包中的函数 ===\n\n")

# 获取所有导出的函数
exported_functions <- ls("package:newrllama4")

cat("📋 导出的函数列表:\n")
for (func in sort(exported_functions)) {
  cat("  -", func, "\n")
}

cat("\n🔍 查找与template相关的函数:\n")
template_functions <- exported_functions[grepl("template", exported_functions, ignore.case = TRUE)]

if (length(template_functions) > 0) {
  for (func in template_functions) {
    cat("  ✅", func, "\n")
  }
} else {
  cat("  ❌ 没有找到与template相关的函数\n")
}

# 检查是否有聊天相关的函数
cat("\n🔍 查找与chat/format相关的函数:\n")
chat_functions <- exported_functions[grepl("chat|format|apply", exported_functions, ignore.case = TRUE)]

if (length(chat_functions) > 0) {
  for (func in chat_functions) {
    cat("  ✅", func, "\n")
  }
} else {
  cat("  ❌ 没有找到与chat/format相关的函数\n")
}

cat("\n📚 所有可用函数:\n")
cat("="*50, "\n")
for (func in sort(exported_functions)) {
  tryCatch({
    func_obj <- get(func, envir = as.environment("package:newrllama4"))
    if (is.function(func_obj)) {
      cat(sprintf("  %s() - 函数\n", func))
    } else {
      cat(sprintf("  %s - %s\n", func, class(func_obj)[1]))
    }
  }, error = function(e) {
    cat(sprintf("  %s - 无法确定类型\n", func))
  })
}