# 部署脚本使用说明

## 概述

`deploy.sh` 是一个自动化脚本，用于构建FPGA教程GitBook并部署到GitHub Pages。

## 前置要求

1. **Node.js** (版本 14 或更高)
2. **GitBook CLI**
3. **Git仓库** (已配置远程仓库)

## 安装GitBook CLI

```bash
# 安装GitBook CLI
npm install -g gitbook-cli

# 获取最新版本的GitBook
gitbook fetch latest
```

## 使用方法

### 1. 仅构建文档

```bash
./deploy.sh
```

这会：
- 检查Node.js和GitBook CLI环境
- 清理之前的构建
- 构建GitBook文档到 `_book/` 目录
- 清理多余文件

### 2. 构建并部署到GitHub Pages

```bash
./deploy.sh deploy
```

这会：
- 执行上述所有构建步骤
- 自动切换到 `gh-pages` 分支
- 推送更新到GitHub
- 切换回 `main` 分支

## 部署后的访问地址

部署成功后，文档将在以下地址可用：
```
https://xde1997.github.io/Tutorial_FPGA/
```

## 故障排除

### GitBook CLI未安装
```bash
npm install -g gitbook-cli
gitbook fetch latest
```

### 构建失败
检查：
1. `book.json` 配置文件是否正确
2. 所有章节文件是否存在
3. Node.js版本是否兼容

### 部署失败
检查：
1. Git仓库是否正确配置
2. 是否有 `gh-pages` 分支
3. 是否有推送权限

## 脚本功能

- ✅ 环境检查 (Node.js, GitBook CLI)
- ✅ 自动构建 GitBook 文档
- ✅ 清理多余文件
- ✅ 自动 Git 操作 (切换分支、提交、推送)
- ✅ 错误处理和用户友好的提示
- ✅ 支持仅构建或构建+部署两种模式

## 注意事项

1. 脚本会自动清理 `_book/` 目录中的以下文件：
   - `deploy.sh`
   - `GITBOOK_SETUP.md`
   - `CONVERSION_SUMMARY.md`

2. 部署时会自动切换到 `gh-pages` 分支，完成后会切换回 `main` 分支

3. 如果 `gh-pages` 分支不存在，脚本会自动创建

4. 确保在运行脚本前已提交所有更改到 `main` 分支 