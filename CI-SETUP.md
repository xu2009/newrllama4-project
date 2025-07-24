# R-CMD-check CI Setup for newrllama4

这个文档详细说明了为 newrllama4 项目设置的 R-CMD-check GitHub Actions 工作流。

## 项目特点

newrllama4 项目有以下特殊性，需要在 CI 中特别处理：

1. **C++ 后端集成**：使用 Rcpp 桥接 R 和 C++ llama.cpp 后端
2. **预编译二进制分发**：依赖从 GitHub Releases 下载的预编译库
3. **动态库加载**：运行时通过 `dlopen` 加载共享库
4. **多平台支持**：支持 macOS (arm64), Windows (x64), Linux (x64)
5. **内存密集型操作**：模型加载和推理需要大量内存

## 工作流设计

### 主要作业 (Jobs)

#### 1. R-CMD-check
标准的 R 包检查，包含：
- 多平台矩阵测试 (macOS, Windows, Linux)
- 多 R 版本支持 (devel, release, oldrel-1)
- 系统依赖安装
- 预编译库预安装
- 标准 R CMD check

#### 2. extended-tests  
扩展功能测试，包含：
- 实际的后端库安装测试
- 基本功能验证
- 更宽松的错误处理

#### 3. package-structure
包结构检查，包含：
- DESCRIPTION 文件验证
- 必要文件存在性检查
- 包元数据加载测试

## 关键配置

### 环境变量

```yaml
env:
  GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}
  R_KEEP_PKG_SOURCE: yes
  _R_CHECK_INTERNET_: TRUE                    # 允许网络访问
  _R_CHECK_TESTS_NLINES_: 0                   # 无限制测试输出行数
  NEWRLLAMA_CACHE_DIR: ${{ runner.temp }}/newrllama_cache
  NEWRLLAMA_TEST_MODE: TRUE                   # CI 测试模式标志
```

### 系统依赖

#### macOS
```bash
brew install cmake
xcode-select --install || true
```

#### Linux  
```bash
sudo apt-get install -y \
  cmake build-essential \
  libcurl4-openssl-dev libssl-dev \
  libxml2-dev libfontconfig1-dev \
  libharfbuzz-dev libfribidi-dev \
  libfreetype6-dev libpng-dev \
  libtiff5-dev libjpeg-dev \
  pciutils
```

#### Windows
使用 Rtools 提供的构建工具。

### 缓存策略

1. **R 包缓存**：基于操作系统和 R 版本的包缓存
2. **预编译库缓存**：基于操作系统和库版本的二进制缓存

## 测试策略

### CI 环境适配

项目包含专门的 CI 测试辅助函数：

```r
# tests/testthat/helper-ci.R
is_ci()                          # 检测 CI 环境
skip_if_ci()                     # 在 CI 中跳过测试
skip_if_no_backend()             # 无后端时跳过
skip_if_no_network()             # 无网络时跳过
skip_if_memory_intensive()       # 跳过内存密集型测试
```

### 测试分层

1. **基础测试** (`test-basic.R`)：
   - 包结构和函数存在性
   - 参数验证
   - 基本配置功能

2. **集成测试** (`test-integration.R`)：
   - 后端安装和初始化
   - 模拟对象测试
   - 网络和文件操作

## 故障排除

### 常见问题

#### 1. 编译错误
**症状**: C++ 编译失败
**解决**: 
- 检查系统依赖是否正确安装
- 确认 Rcpp 版本兼容性
- 查看 CMake 配置是否正确

#### 2. 预编译库下载失败
**症状**: `install_newrllama()` 失败
**解决**:
- 检查网络连接
- 确认 GitHub Releases 中存在对应平台的文件
- 验证下载 URL 正确性

#### 3. 动态库加载失败
**症状**: `dlopen` 错误
**解决**:
- 检查库文件权限
- 确认库文件完整性
- 验证平台兼容性

#### 4. 内存不足
**症状**: 测试中 OOM 错误
**解决**:
- 跳过内存密集型测试
- 使用模拟对象替代真实模型
- 增加交换空间配置

### 调试技巧

#### 1. 启用详细日志
```yaml
env:
  _R_CHECK_ALL_NON_ISO_C_: TRUE
  R_ENABLE_JIT: 0
```

#### 2. 保留构建工件
```yaml
- name: Upload check results
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: check-results-${{ matrix.config.os }}
    path: check/
```

#### 3. 交互式调试
在工作流中添加：
```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
```

## 最佳实践

### 1. 渐进式部署
建议按以下顺序启用检查：
1. 首先在单一平台测试
2. 逐步添加其他平台
3. 最后启用所有 R 版本

### 2. 错误处理
- 对网络依赖操作使用 `continue-on-error: true`
- 在测试中优雅处理缺失依赖
- 提供清晰的错误消息

### 3. 性能优化
- 使用合适的缓存策略
- 并行运行独立的检查
- 预安装常用依赖

### 4. 维护性
- 定期更新 action 版本
- 监控构建时间变化
- 及时处理已知问题

## 配置文件总览

创建的文件列表：
```
.github/workflows/R-CMD-check.yml      # 主工作流文件
newrllama4/tests/testthat/helper-ci.R  # CI 测试辅助函数
newrllama4/tests/testthat/test-basic.R # 基础测试
newrllama4/tests/testthat/test-integration.R # 集成测试  
newrllama4/.Rbuildignore               # 构建忽略文件
```

## 手动触发

可以通过以下方式手动触发工作流：

1. **推送到主分支**：自动触发所有检查
2. **创建 Pull Request**：自动触发基础检查
3. **添加标签**：为 PR 添加 `extended-tests` 标签触发扩展测试

## 监控和维护

- 定期检查构建状态
- 监控构建时间变化
- 及时更新依赖版本
- 根据新的 R 版本调整矩阵配置

这个 CI 配置专门为 newrllama4 项目的特殊需求而设计，在保证代码质量的同时，充分考虑了 C++ 集成、预编译分发、和多平台兼容性的要求。