# Data Flow Application

这是一个基于 React 和 Tailwind CSS 的数据流管理界面，根据 Figma 设计实现。

## 功能特性

- **数据上传**：支持本地上传和 SharePoint 文件夹上传
- **数据暂存区**：显示已上传文件列表，包含文件名、类型、标签和日期
- **工作流步骤**：清晰的进度指示器（上传 → 清洗 → 处理 → 可视化）
- **文件管理**：支持查看和管理上传的文件

## 安装

1. 安装依赖：
```bash
npm install
```

## 运行

启动开发服务器：
```bash
npm start
```

应用将在 [http://localhost:3000](http://localhost:3000) 打开。

## 构建

创建生产版本：
```bash
npm run build
```

构建文件将生成在 `build` 文件夹中。

## 技术栈

- **React 18**: UI 框架
- **Tailwind CSS 3**: 样式框架
- **React Scripts**: 构建工具

## 项目结构

```
1st demo/
├── public/
│   └── index.html          # HTML 模板
├── src/
│   ├── Input.jsx           # 主组件
│   ├── index.js            # 入口文件
│   └── index.css           # 全局样式
├── package.json            # 项目配置
├── tailwind.config.js      # Tailwind 配置
└── postcss.config.js       # PostCSS 配置
```

## 组件说明

### Input 组件
主要的数据上传和管理界面，包含：
- 顶部导航栏
- 工具栏
- 进度步骤指示器
- 上传卡片（本地/SharePoint）
- 数据暂存区表格
- 上传完成按钮

## 自定义

### 修改颜色
在 [tailwind.config.js](tailwind.config.js) 中扩展主题配置。

### 添加新功能
编辑 [src/Input.jsx](src/Input.jsx) 文件来添加新的功能和交互。

## 注意事项

- 确保已安装 Node.js (推荐 v14 或更高版本)
- 图标使用内联 SVG 实现，无需额外的图标库
- 示例数据在组件中硬编码，实际使用时需要连接后端 API

## License

MIT
