# NMI (Nature Machine Intelligence) 投稿格式检查清单

> **期刊**：Nature Machine Intelligence
> **更新时间**：2024-12-14
> **用途**：确保投稿材料符合期刊格式要求

---

## 📄 文稿格式要求

### 基本文档规范
- [ ] **文件格式**：LaTeX (.tex) 或 Word (.docx)
- [ ] **字体**：Times New Roman
- [ ] **主文本字号**：12 pt
- [ ] **行间距**：单倍行距
- [ ] **页边距**：默认（1 inch / 2.54 cm）
- [ ] **页码**：连续编号

### 文章结构
- [ ] **标题页**：包含标题、作者、单位、摘要
- [ ] **摘要**：≤ 250 words，结构化（Background, Methods, Results, Conclusion）
- [ ] **正文**：
  - Introduction（≤ 1000 words）
  - Methods（≤ 1500 words）
  - Results（≤ 1500 words）
  - Discussion/Conclusion（≤ 800 words）
- [ ] **参考文献**：≤ 50 篇，使用 Nature 风格
- [ ] **致谢**：简短，包含基金信息

---

## 📊 图表格式要求

### 通用要求
- [ ] **图表编号**：按文中出现顺序编号（Fig. 1, Fig. 2...）
- [ ] **图表标题**：简洁明了，置于图表下方
- [ ] **字体大小**：图表内文字 8-10 pt
- [ ] **颜色**：支持色盲友好的配色方案
- [ ] **矢量格式**：优先使用 .pdf, .eps, .svg

### 单栏图（Single-column）
- [ ] **宽度**：≤ 89 mm（3.5 inches）
- [ ] **分辨率**：≥ 300 dpi
- [ ] **适用场景**：简单图表、流程图
- [ ] **文件格式**：.pdf（矢量图优先）

### 双栏图（Double-column）
- [ ] **宽度**：≤ 183 mm（7.2 inches）
- [ ] **分辨率**：≥ 600 dpi
- [ ] **适用场景**：复杂示意图、多面板图
- [ ] **文件格式**：.pdf 或 .eps

### 表格格式
- [ ] **格式**：使用 LaTeX table 环境
- [ ] **标题**：置于表格上方（Table 1: ...）
- [ ] **内容**：避免垂直线，使用水平线分隔
- [ ] **字体**：可适当缩小至 10 pt
- [ ] **引用**：每个表格需在正文中引用

---

## 🎨 具体图表检查清单

### Fig. 1: 模型架构图
- [ ] **尺寸**：双栏（183 mm）
- [ ] **分辨率**：600 dpi
- [ ] **内容要求**：
  - 1D 分支结构清晰
  - 2D 分支结构清晰
  - 融合机制标注
  - 对齐模块位置
- [ ] **文件格式**：.pdf
- [ ] **颜色**：确保黑白打印可读

### Fig. 2: 融合机制示意图
- [ ] **尺寸**：双栏（183 mm）
- [ ] **分辨率**：600 dpi
- [ ] **子图数量**：≤ 4 个
- [ ] **标注**：A, B, C, D 标记清晰
- [ ] **说明文字**：图注中详细说明

### Fig. 3: 性能对比结果
- [ ] **尺寸**：单栏（89 mm）
- [ ] **分辨率**：300 dpi
- [ ] **图表类型**：柱状图或箱线图
- [ ] **误差线**：标准差或置信区间
- [ ] **显著性标记**：p-value 或星号标记

### Fig. 4: 可视化与解释
- [ ] **尺寸**：双栏（183 mm）
- [ ] **分辨率**：600 dpi
- [ ] **热力图**：color bar 标注清晰
- [ ] **归因图**：梯度方向明确
- [ ] **对比图**：包含基线方法

---

## 📑 LaTeX 模板检查

### 必需的 LaTeX 包
```latex
\usepackage{graphicx}      % 插入图片
\usepackage{amsmath}       % 数学公式
\usepackage{natbib}        % 参考文献
\usepackage{url}           % URL链接
\usepackage{hyperref}      % 超链接
\usepackage{booktabs}      % 表格格式
\usepackage{multirow}      % 多行表格
\usepackage{multicol}      % 多栏布局
```

### 文档类设置
```latex
\documentclass[12pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
```

### 参考文献格式
```latex
\bibliographystyle{naturemag}
\bibliography{references}
```

---

## 📁 文件组织规范

### 主文件夹结构
```
submission/
├── manuscript.tex          # 主文稿
├── figures/               # 图文件夹
│   ├── fig1_architecture.pdf
│   ├── fig2_fusion.pdf
│   ├── fig3_results.pdf
│   └── fig4_visualization.pdf
├── tables/                # 表格文件夹
│   ├── table1_performance.tex
│   └── table2_ablation.tex
├── supplementary/         # 补充材料
│   ├── supplementary.pdf
│   └── suppl_figures/
├── references.bib         # 参考文献数据库
└── cover_letter.pdf       # 投稿信
```

### 文件命名规范
- [ ] 主图：`fig{N}_{description}.pdf`
- [ ] 补充图：`fig{N}s_{description}.pdf`
- [ ] 表格：`table{N}_{description}.tex`
- [ ] 补充材料：`supplementary.pdf`

---

## ✅ 提交前最终检查

### PDF 生成检查
- [ ] **PDF 版本**：1.4 或更高
- [ ] **文件大小**：≤ 20 MB（主文稿）
- [ ] **所有字体嵌入**：File > Properties > Fonts
- [ ] **书签生成**：章节导航
- [ ] **超链接**：参考文献可点击

### 内容完整性
- [ ] **所有图表**：已插入正确位置
- [ ] **所有引用**：参考文献编号正确
- [ ] **作者信息**：ORCID iDs 完整
- [ ] **基金信息**：Grant numbers 正确
- [ ] **利益声明**：Competing interests

### 快速验证命令
```bash
# 检查 PDF 大小
ls -lh manuscript.pdf

# 检查字体嵌入
pdffonts manuscript.pdf

# 检查 PDF 版本
pdfinfo manuscript.pdf

# 统计字数
texcount -sum manuscript.tex
```

---

## 🚨 常见问题与解决方案

### 图表问题
1. **分辨率不足**：使用矢量格式重绘
2. **文件过大**：压缩图片，优化 PDF
3. **字体显示**：确保所有系统字体嵌入

### LaTeX 问题
1. **编译错误**：逐行检查，查看 .log 文件
2. **参考文献问题**：检查 .bib 文件格式
3. **图片路径**：使用相对路径

### 格式问题
1. **超限提醒**：缩减文字或申请豁免
2. **图注过长**：移部分内容到正文
3. **表格过宽**：调整列宽或旋转表格

---

## 📞 获取帮助

- **期刊指南**：https://www.nature.com/natmachintel/authors-and-referees
- **LaTeX 模板**：期刊官网下载最新版本
- **客服支持**：nature@nature.com

---

## ✅ 提交清单

提交前请确认：
- [ ] 所有格式要求已满足
- [ ] PDF 文件正常生成
- [ ] 所有必需文件已准备
- [ ] 补充材料完整
- [ ] 投稿信已撰写
- [ ] 作者已最终确认

---

**最后更新**：2024-12-14
**版本**：v1.0
**下次检查**：提交前1天