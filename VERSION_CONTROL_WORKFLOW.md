# newrllama4 版本控制和发布流程

本文档详细描述了 newrllama4 项目的版本控制流程，确保代码修改、版本管理和自动化发布的一致性。

## 1. 项目架构概述

newrllama4 采用四层分离架构：
- **开发层**: `custom_files/` - 主要开发文件
- **编译层**: `backend/llama.cpp/` - 实际编译位置
- **R包层**: `newrllama4/` - R包源码
- **分发层**: GitHub Releases - 预编译库分发

## 2. 当前版本状态

**当前版本**: 1.0.62
- R包版本: `newrllama4/DESCRIPTION` → Version: 1.0.62
- 库版本: `newrllama4/R/install.R` → .lib_version <- "1.0.62"
- 下载URL: `newrllama4/R/install.R` → .base_url <- "https://github.com/xu2009/newrllama4-project/releases/download/v1.0.62/"

## 3. 完整版本发布流程

### 步骤1：修改核心文件

修改以下主要开发文件：
```bash
# 主要C++实现文件
custom_files/newrllama_capi.cpp

# C-API接口定义（如需要）
custom_files/newrllama_capi.h
```

### 步骤2：同步文件到编译位置

将修改同步到实际编译位置：
```bash
# 同步C++实现
cp custom_files/newrllama_capi.cpp backend/llama.cpp/newrllama_capi.cpp

# 同步头文件
cp custom_files/newrllama_capi.h backend/llama.cpp/newrllama_capi.h
```

### 步骤3：更新版本号

假设要发布版本 1.0.63，需要在以下文件中统一更新版本号：

#### 3.1 R包安装配置
```bash
# 文件: newrllama4/R/install.R
# 从：
.lib_version <- "1.0.62"
.base_url <- "https://github.com/xu2009/newrllama4-project/releases/download/v1.0.62/"

# 改为：
.lib_version <- "1.0.63"
.base_url <- "https://github.com/xu2009/newrllama4-project/releases/download/v1.0.63/"
```

#### 3.2 R包描述文件
```bash
# 文件: newrllama4/DESCRIPTION
# 从：
Version: 1.0.62

# 改为：
Version: 1.0.63
```

### 步骤4：版本一致性检查

运行检查命令确保版本号统一：
```bash
# 检查C++文件是否同步
diff custom_files/newrllama_capi.cpp backend/llama.cpp/newrllama_capi.cpp

# 检查版本号是否统一（更新搜索模式为新版本）
grep -r "1\.0\.63" newrllama4/ --include="*.R" --include="DESCRIPTION"
```

### 步骤5：提交代码更改

```bash
# 添加所有更改
git add .

# 提交更改（使用描述性消息）
git commit -m "Release v1.0.63: [具体修复或功能描述]"

# 推送到主分支
git push origin master
```

### 步骤6：创建发布标签

```bash
# 创建版本标签
git tag v1.0.63

# 推送标签到远程
git push origin v1.0.63
```

### 步骤7：等待自动化构建

GitHub Actions 会自动：
1. 检测到新的 release tag
2. 触发多平台并行构建 (`.github/workflows/release-builder.yml`)
3. 编译预编译库：
   - `libnewrllama_linux_x64.zip`
   - `newrllama_windows_x64.zip`  
   - `libnewrllama_macos_x64.zip`
   - `libnewrllama_macos_arm64.zip`
4. 上传到 GitHub Releases

### 步骤8：验证发布

检查以下内容：
- [ ] GitHub Actions 构建成功
- [ ] Release 页面包含所有4个预编译库
- [ ] 版本号正确 (v1.0.63)
- [ ] 文件大小合理（通常 ~1MB）

### 步骤9：本地测试

```bash
# 重新安装最新版本
install_newrllama()

# 运行基本功能测试
model <- model_load("path/to/test/model.gguf")
response <- quick_llama("Hello world")
```

## 4. 关键注意事项

### 4.1 文件同步要求
- **必须**: 将 `custom_files/` 中的更改同步到 `backend/llama.cpp/`
- **原因**: GitHub Actions 从 `backend/llama.cpp/` 目录进行编译

### 4.2 版本号一致性
- 所有版本号必须完全一致
- 包括 URL 中的版本标签
- 不一致会导致下载失败

### 4.3 构建平台支持
当前支持的平台：
- **Linux**: x86_64 (Ubuntu latest)
- **Windows**: x86_64 (静态链接，vcpkg)
- **macOS**: x86_64 (Intel) 和 arm64 (Apple Silicon)

### 4.4 常见错误排查

#### 下载失败
```bash
# 检查URL是否正确（用实际版本号替换）
curl -I https://github.com/xu2009/newrllama4-project/releases/download/v1.0.63/libnewrllama_linux_x64.zip
```

#### 版本不匹配
```bash
# 检查R包中的版本配置
grep -n "1\.0\." newrllama4/R/install.R
grep -n "Version:" newrllama4/DESCRIPTION
```

#### 构建失败
1. 检查 GitHub Actions 日志
2. 确认 custom files 已正确同步
3. 验证 CMake 配置正确

## 5. 发布检查清单

发布前确认：
- [ ] 核心文件已修改并测试
- [ ] 文件已同步到编译位置 (`cp custom_files/* backend/llama.cpp/`)
- [ ] 版本号在所有文件中统一更新
- [ ] 代码已提交到 master 分支
- [ ] 创建并推送了版本标签
- [ ] GitHub Actions 构建成功
- [ ] Release 页面包含所有预编译库
- [ ] 本地测试通过

## 6. 回滚流程

如发现问题需要回滚：

```bash
# 删除有问题的标签
git tag -d v1.0.XX
git push origin :refs/tags/v1.0.XX

# 在 GitHub 上删除对应的 Release
# 修复问题后重新发布
```

## 7. 版本号规则

- 使用语义化版本控制：`MAJOR.MINOR.PATCH`
- 当前使用 1.0.x 系列
- Bug修复：递增PATCH版本 (1.0.62 → 1.0.63)
- 新功能：递增MINOR版本 (1.0.x → 1.1.0)
- 破坏性更改：递增MAJOR版本 (1.x.x → 2.0.0)

---

遵循此流程可确保 newrllama4 项目的版本管理和发布过程稳定、可追溯和自动化。