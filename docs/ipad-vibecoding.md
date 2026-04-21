# iPad Vibecoding 工作流

这份指南用于把 iPad 作为“轻开发终端”：**改代码 + 管理 Git + 触发远程运行**。

## 目标

- 少输入：统一用 `make` 别名
- 少切换：Git 操作集中在 Working Copy
- 少踩坑：把高负载任务放在远程 Linux/GPU 机器

## 推荐软件栈

1. **Working Copy**（必备）
   - clone 仓库
   - 分支管理、commit、push、pull request
2. **Blink Shell** 或 **a-Shell**（建议）
   - SSH 到云主机 / 家里 PC
   - 执行训练、推理、测试命令
3. **浏览器（Safari）**（可选）
   - 打开 GitHub/Codespaces/自建 code-server

## 一次性准备

### A. 在远程机器准备环境

```bash
git clone <your-repo-url>
cd my-vibecoding-project
make setup
```

### B. iPad 端建议设置

- Working Copy 开启：
  - 自动 fetch
  - 显示未跟踪文件
- SSH 工具保存 host 别名（例如 `gpu-main`）
- 常用片段保存为文本替换（如 `;gp` => `git pull`）

## 每日流程（推荐）

1. **拉取**：Working Copy `Pull`。
2. **改动**：在 iPad 编辑（优先文档、配置、脚本、小改动）。
3. **验证**：SSH 到远程仓库目录运行：
   ```bash
   make check
   make demo-infer
   ```
4. **提交**：Working Copy 写清楚 commit message。
5. **同步**：Push 后在 GitHub 发起 PR。

## 针对本仓库的快捷命令

在仓库根目录可直接执行：

```bash
make help
make setup
make check
make demo-infer
make demo-app
```

说明：

- `make setup`：安装 `ccpd_alpr` 依赖并 editable 安装
- `make check`：做基础语法检查（不跑重训练）
- `make demo-infer`：跑 demo 图片推理
- `make demo-app`：启动可视化 app

## iPad 使用边界（实话实说）

- 大规模训练请在远程机器执行
- iPad 更适合：
  - 写文档
  - 调配置
  - 小功能 patch
  - review 与提交

## 建议的分支策略

- `main`：稳定版本
- `feat/ipad-*`：移动端改进
- `exp/*`：实验脚本

这样在 iPad 上切分支不会混乱，也更容易回滚。
