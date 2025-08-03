#!/bin/bash

# FPGA教程GitBook部署脚本
# 用于构建GitBook并部署到GitHub Pages

set -e

echo "🚀 开始构建FPGA教程GitBook..."

# 检查Node.js版本
check_node_version() {
    local node_version=$(node --version 2>/dev/null | sed 's/v//')
    if [ -z "$node_version" ]; then
        echo "❌ Node.js未安装"
        exit 1
    fi
    
    echo "✅ 使用Node.js v$node_version"
}

# 构建文档
build_docs() {
    echo "🔨 构建文档..."
    
    # 检查GitBook是否可用
    if ! command -v gitbook &> /dev/null; then
        echo "❌ GitBook不可用"
        echo "💡 请先安装GitBook CLI："
        echo "   npm install -g gitbook-cli@2.3.0"
        exit 1
    fi
    
    # 清理之前的构建
    echo "🧹 清理之前的构建..."
    rm -rf _book
    
    # 使用GitBook构建
    echo "📦 使用GitBook构建..."
    echo "🔍 执行构建命令: gitbook build . _book"
    
    # 执行构建命令并捕获输出
    build_output=$(gitbook build . _book 2>&1)
    build_exit_code=$?
    
    echo "📋 构建输出:"
    echo "$build_output"
    
    # 检查构建是否成功
    if [ $build_exit_code -eq 0 ] && [ -d "_book" ]; then
        echo "✅ GitBook构建成功"
        
        # 验证关键文件是否存在
        if [ -f "_book/index.html" ]; then
            echo "✅ 验证: index.html 文件存在"
        else
            echo "❌ 验证失败: index.html 文件不存在"
            return 1
        fi
        
        # 清理多余文件
        echo "🧹 清理多余文件..."
        rm -f _book/deploy.sh _book/docs/GITBOOK_SETUP.md _book/docs/CONVERSION_SUMMARY.md 2>/dev/null || true
        
        return 0
    else
        echo "❌ GitBook构建失败"
        echo "💡 构建退出代码: $build_exit_code"
        echo "💡 请检查："
        echo "   1. GitBook CLI是否正确安装"
        echo "   2. book.json配置文件是否正确"
        echo "   3. 所有章节文件是否存在"
        echo "   4. Node.js版本是否兼容"
        return 1
    fi
}

# 部署到GitHub Pages
deploy_to_github() {
    if [ "$1" = "deploy" ]; then
        echo "🚀 部署到GitHub Pages..."
        
        # 设置部署分支（默认为gh-pages，可通过第二个参数指定）
        local deploy_branch="${2:-gh-pages}"
        echo "📝 部署分支: $deploy_branch"
        
        # 检查Git仓库位置
        local git_dir=""
        if [ -d ".git" ]; then
            git_dir="."
        elif [ -d "../.git" ]; then
            git_dir=".."
        else
            echo "❌ 当前目录及其上级目录都不是Git仓库"
            exit 1
        fi
        
        echo "📁 使用Git仓库: $git_dir"
        
        # 切换到Git仓库目录
        cd "$git_dir"
        
        # 复制文档到临时目录
        echo "📋 准备文档..."
        if [ "$git_dir" = "." ]; then
            cp -r _book ./_book_temp
        else
            cp -r doc/_book ./_book_temp
        fi
        
        # 检查目标分支是否存在，如果不存在则创建
        if ! git show-ref --verify --quiet refs/heads/$deploy_branch; then
            echo "📝 创建新分支: $deploy_branch"
            git checkout -b $deploy_branch
        else
            echo "📝 切换到分支: $deploy_branch"
            git checkout $deploy_branch
        fi
        
        # 清理当前分支内容（保留.git目录）
        echo "🧹 清理当前分支内容..."
        find . -mindepth 1 -not -path './.git*' -delete
        
        # 复制文档内容
        echo "📋 复制文档内容..."
        cp -r _book_temp/* .
        rm -rf _book_temp
        
        # 提交更改
        echo "📤 提交更改..."
        git add .
        git commit -m "Update documentation $(date)" || {
            echo "⚠️  没有新的更改需要提交"
        }
        
        # 推送到远程仓库
        echo "🚀 推送到GitHub..."
        git push origin $deploy_branch
        
        # 切换回main分支
        git checkout main
        
        # 返回到doc目录
        if [ "$git_dir" = ".." ]; then
            cd doc
        fi
        
        echo "✅ 部署完成！"
        echo "🌐 文档将在几分钟后可在以下地址访问："
        echo "   https://xde1997.github.io/Tutorial_FPGA/"
        echo "💡 如果使用自定义分支，请在GitHub仓库设置中配置GitHub Pages源分支为: $deploy_branch"
    else
        echo "📖 文档已构建完成，位于 _book/ 目录"
        echo "💡 运行 './deploy.sh deploy' 来部署到GitHub Pages"
        echo "💡 运行 './deploy.sh deploy gp_graph' 来部署到自定义分支"
    fi
}

# 主执行流程
main() {
    # 检查Node.js版本
    check_node_version
    
    # 构建文档
    if ! build_docs; then
        echo "❌ 构建失败，停止执行"
        exit 1
    fi
    
    # 检查构建是否成功
    if [ ! -d "_book" ]; then
        echo "❌ 构建失败：_book目录不存在"
        exit 1
    fi
    
    echo "✅ 文档构建成功！"
    
    # 部署
    deploy_to_github "$1"
    
    echo "🎉 完成！"
}

# 错误处理
trap 'echo "❌ 脚本执行被中断"; exit 1' INT TERM

# 执行主函数
main "$@" 