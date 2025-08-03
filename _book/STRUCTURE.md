# 文档结构说明

## 目录结构

```
fpga_tutorial/
├── README.md              # 首页
├── SUMMARY.md             # 目录导航
├── book.json              # GitBook配置
├── .gitbook.yaml          # 现代GitBook配置
├── deploy.sh              # 自动化部署脚本
├── .gitignore             # Git忽略文件
├── STRUCTURE.md           # 本文件
├── chapters/              # 章节文件目录
│   ├── chapter1.md        # 第1章：FPGA基础架构与工作原理
│   ├── chapter2.md        # 第2章：HDL设计基础与方法学
│   ├── chapter3.md        # 第3章：时序、时钟与同步
│   ├── chapter4.md        # 第4章：存储器系统与接口设计
│   ├── chapter5.md        # 第5章：高速I/O与通信
│   ├── chapter6.md        # 第6章：DSP与算术优化
│   ├── chapter7.md        # 第7章：HLS与C到硬件综合
│   ├── chapter8.md        # 第8章：函数式HDL之Haskell/Clash
│   ├── chapter9.md        # 第9章：OCaml/Hardcaml硬件设计
│   ├── chapter10.md       # 第10章：零知识证明加速器
│   ├── chapter11.md       # 第11章：AI加速器基础
│   ├── chapter12.md       # 第12章：LLM推理加速
│   ├── chapter13.md       # 第13章：视觉与多模态处理
│   ├── chapter14.md       # 第14章：LLM服务基础设施
│   ├── chapter15.md       # 第15章：机器人运动控制与FPGA
│   ├── chapter16.md       # 第16章：激光雷达信号处理与FPGA
│   ├── chapter17.md       # 第17章：毫米波雷达与FPGA
│   ├── chapter18.md       # 第18章：性能分析与优化
│   ├── chapter19.md       # 第19章：功耗优化技术
│   ├── chapter20.md       # 第20章：多FPGA系统与扩展
│   ├── chapter21.md       # 第21章：可靠性与容错设计
│   └── chapter22.md       # 第22章：未来趋势与新兴技术
├── docs/                  # 文档说明目录
│   ├── GITBOOK_SETUP.md   # GitBook安装使用指南
│   ├── DEPLOY_README.md   # 部署脚本使用说明
│   └── CONVERSION_SUMMARY.md # 转换完成总结
├── assets/                # 资源文件目录
│   ├── images/            # 图片文件
│   ├── css/               # 自定义样式
│   └── js/                # 自定义脚本
└── node_modules/          # Node.js依赖（自动生成）
```

## 文件说明

### 核心文件
- **README.md**: GitBook首页，包含项目介绍和学习路径
- **SUMMARY.md**: 目录导航文件，定义章节结构和链接
- **book.json**: GitBook配置文件，包含主题、插件等设置
- **.gitbook.yaml**: 现代GitBook配置文件

### 章节文件
所有章节文件都放在 `chapters/` 目录下，便于管理和维护。

### 文档说明
- **GITBOOK_SETUP.md**: GitBook安装和使用指南
- **DEPLOY_README.md**: 部署脚本的详细使用说明
- **CONVERSION_SUMMARY.md**: 从普通Markdown转换为GitBook的总结

### 资源文件
- **assets/**: 存放图片、CSS、JavaScript等资源文件
- **images/**: 图片文件目录
- **css/**: 自定义样式文件
- **js/**: 自定义脚本文件

## 使用说明

### 添加新章节
1. 在 `chapters/` 目录下创建新的 `.md` 文件
2. 在 `SUMMARY.md` 中添加对应的链接
3. 确保章节文件有正确的标题结构

### 添加图片
1. 将图片文件放在 `assets/images/` 目录下
2. 在Markdown中使用相对路径引用：`![描述](assets/images/图片名.png)`

### 自定义样式
1. 在 `assets/css/` 目录下创建CSS文件
2. 在 `book.json` 中引用CSS文件

## 构建和部署

### 本地构建
```bash
./deploy.sh
```

### 构建并部署
```bash
./deploy.sh deploy
```

### 验证文档结构
```bash
# 检查所有文件和目录是否正确
ls -la
ls chapters/ | wc -l  # 应该显示22个章节文件
```

## 注意事项

1. 所有章节文件必须放在 `chapters/` 目录下
2. 图片等资源文件放在 `assets/` 目录下
3. 更新章节后需要重新构建文档
4. 部署前确保所有更改已提交到Git 