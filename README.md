# my-vibecoding-project
<<<<<<< ours

我的 vibecoding 专属练习与代码库。现在已按“**iPad 轻量协作**”方式整理，方便用移动端工具快速浏览、编辑、提交和运行。  

## 目录结构（已整理）

- `README.md`：仓库总览（你现在看到的这个文件）
- `docs/ipad-vibecoding.md`：iPad 端工作流说明（推荐先看）
- `Makefile`：常用命令快捷入口（减少在 iPad 上敲长命令）
- `ccpd_alpr/`：车牌识别主项目（训练、推理、UI）

## 快速开始（iPad 友好）

### 1) 推荐软件组合

- **Working Copy**：Git 拉取/提交/分支管理
- **Blink Shell / a-Shell**：SSH 到远程 Linux 机器执行训练与推理
- （可选）**VS Code Web / GitHub Codespaces**：浏览器内编辑

> iPad 本地不建议做大模型训练，建议作为“控制台 + 提交端”。

### 2) 常用命令（在仓库根目录执行）

```bash
make help
make setup
make demo-infer
make demo-app
```

### 3) 常规工作流

1. Working Copy 拉取最新代码
2. iPad 编辑文档/脚本
3. 通过 `make` 命令在远程机器验证
4. 提交并推送

---

## 主项目入口

详见：`ccpd_alpr/README.md`
=======
我的 vibecoding 专属练习与代码库

## 新增项目：Python 五子棋人机对战

已在 `gomoku_dl/` 下实现一个基于 Python 的五子棋 GUI 项目：

- `gomoku_dl/gomoku_game.py`：完整游戏与 AI 实现
- `gomoku_dl/README.md`：运行说明与功能列表

运行：

```bash
python gomoku_dl/gomoku_game.py
```
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
