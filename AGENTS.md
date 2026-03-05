# Utonia 开发者 / Agent 指南（中文）

本仓库用于 **Utonia: Toward One Encoder for All Point Clouds** 的推理代码、可视化 demo 与预训练权重使用示例。

## 仓库结构
- `utonia/`：核心 Python 包（模型、结构体、数据与 transform）。
- `demo/`：可视化与推理示例脚本（见 `README.md` 的 Quick Start）。
- `weights/`：本地权重（通常被 `.gitignore` 忽略，不要提交新增大文件）。
- `environment.yml`：Conda 环境（含 CUDA / PyTorch 版本约束）。
- `setup.py`：安装为 Python 包（不负责自动安装 CUDA 相关依赖）。

## 常用命令
- **Conda（推荐用于跑 demo）**
  - `conda env create -f environment.yml --verbose`
  - `conda activate utonia`
- **运行 demo**
  - `export PYTHONPATH=./`
  - `python demo/0_pca_indoor.py`（其余脚本见 `README.md`）

## 权重与缓存
- **从 HuggingFace 下载并缓存**
  - `utonia.model.load("utonia", repo_id="Pointcept/Utonia")`
  - 默认缓存目录：`~/.cache/utonia/ckpt`（可通过 `download_root` 参数自定义）
- **从本地权重加载**
  - `utonia.model.load("weights/utonia.pth")`
- **不要提交数据/权重/输出**
  - `.gitignore` 已忽略 `weights/`, `data/`, `ckpt/`, `outputs/`, `log/` 等目录；请保持该约定。

## 修改约定（建议）
- 尽量保持改动小且聚焦，避免无意义的全文件格式化/重排 import。
- 新增示例或可运行入口时，优先放在 `demo/`，并保持与现有脚本一致的参数与依赖风格。
- 依赖包含 CUDA 扩展（如 `spconv` / `flash-attn` / `torch-scatter`），在无 GPU 环境可能无法运行；提交变更时至少确保 `import utonia` 不报错，并在说明中标注是否需要 GPU 才能复现。

## 快速自检（仓库无单元测试时）
- 仅验证可导入：`python -c "import utonia; print('import ok')"`
- 需要完整依赖与 GPU 的烟测：`python demo/3_batch_forward.py`

