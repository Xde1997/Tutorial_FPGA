# GitBook 安装与使用指南

本教程已转换为GitBook格式，以下是安装和使用说明。

## 安装 GitBook

### 方法一：使用 Node.js (推荐)

1. 安装 Node.js (版本 14 或更高)
   ```bash
   # 从 https://nodejs.org 下载并安装
   ```

2. 安装 GitBook CLI
   ```bash
   npm install -g gitbook-cli
   ```

3. 安装 GitBook
   ```bash
   gitbook fetch latest
   ```

### 方法二：使用 Docker

```bash
docker run --rm -p 4000:4000 -v $(pwd):/gitbook billryan/gitbook gitbook serve
```

## 本地运行

1. 进入项目目录
   ```bash
   cd fpga_tutorial
   ```

2. 安装依赖
   ```bash
   gitbook install
   ```

3. 启动本地服务器
   ```bash
   gitbook serve
   ```

4. 在浏览器中访问
   ```
   http://localhost:4000
   ```

## 构建静态网站

```bash
gitbook build
```

生成的静态文件将保存在 `_book` 目录中。

## 部署到 GitHub Pages

### 方法一：使用自动化脚本（推荐）

项目提供了自动化的部署脚本 `deploy.sh`：

1. **仅构建文档**：
   ```bash
   ./deploy.sh
   ```

2. **构建并部署到GitHub Pages**：
   ```bash
   ./deploy.sh deploy
   ```

脚本会自动：
- 检查Node.js环境
- 构建GitBook文档
- 清理多余文件
- 切换到gh-pages分支
- 推送更新到GitHub

### 方法二：手动部署

1. 构建静态文件
   ```bash
   gitbook build
   ```

2. 将 `_book` 目录的内容推送到 GitHub Pages 分支
   ```bash
   cd _book
   git init
   git add .
   git commit -m "Deploy to GitHub Pages"
   git branch -M gh-pages
   git remote add origin https://github.com/Xde1997/Tutorial_FPGA.git
   git push -u origin gh-pages
   ```

3. 在 GitHub 仓库设置中启用 GitHub Pages，选择 `gh-pages` 分支

## 文件结构说明

```
fpga_tutorial/
├── README.md              # 首页
├── SUMMARY.md             # 目录导航
├── book.json              # GitBook 配置
├── .gitbook.yaml          # 现代 GitBook 配置
├── deploy.sh              # 自动化部署脚本
├── GITBOOK_SETUP.md       # 安装使用指南
├── CONVERSION_SUMMARY.md  # 转换完成总结
├── chapter1.md            # 第1章
├── chapter2.md            # 第2章
├── ...
└── chapter22.md           # 第22章
```

## 配置说明

### book.json
- `title`: 书籍标题
- `description`: 书籍描述
- `language`: 语言设置 (zh-hans)
- `plugins`: 启用的插件列表
- `pluginsConfig`: 插件配置

### 主要插件
- `expandable-chapters`: 可折叠章节
- `search`: 搜索功能
- `copy-code-button`: 代码复制按钮
- `back-to-top-button`: 返回顶部按钮
- `github`: GitHub 链接
- `toc`: 目录导航

## 自定义主题

可以通过修改 `book.json` 中的 `variables` 部分来自定义主题颜色：

```json
{
  "variables": {
    "themeColor": "#3f51b5",
    "themeColorSecondary": "#ff4081"
  }
}
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   gitbook serve --port 4001
   ```

2. **插件安装失败**
   ```bash
   gitbook install --force
   ```

3. **构建失败**
   ```bash
   gitbook build --log debug
   ```

### 版本兼容性

- GitBook CLI: 2.3.2 或更高
- Node.js: 14.x 或更高
- 推荐使用 LTS 版本

## 更新内容

1. 修改相应的 `.md` 文件
2. 重新构建
   ```bash
   gitbook build
   ```
3. 重新部署到服务器

## 贡献指南

1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

---

更多信息请参考 [GitBook 官方文档](https://toolchain.gitbook.com/)。 