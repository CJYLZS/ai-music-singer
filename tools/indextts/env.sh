# source 我，给 IndexTTS-2.5 用的隔离环境变量。
# 所有 cache 都必须落在 workspace 里：DSH 沙箱下 ~/.cache 与 ~/.config 是只读的。
IDX_ROOT=/home/rookie/voice/tools/indextts

# uv
export UV_CACHE_DIR="$IDX_ROOT/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$IDX_ROOT/.uv-python"
export UV_PROJECT_ENVIRONMENT="$IDX_ROOT/.venv"

# 模型权重 / 各家 SDK 缓存
export MODELSCOPE_CACHE="$IDX_ROOT/.ms-cache"
export HF_HOME="$IDX_ROOT/.hf-home"
export TORCH_HOME="$IDX_ROOT/.torch-home"
export XDG_CACHE_HOME="$IDX_ROOT/.xdg-cache"
export XDG_CONFIG_HOME="$IDX_ROOT/.xdg-config"
export MPLCONFIGDIR="$IDX_ROOT/.mpl"
export NUMBA_CACHE_DIR="$IDX_ROOT/.numba-cache"
export TRITON_CACHE_DIR="$IDX_ROOT/.triton-cache"
export TOKENIZERS_PARALLELISM=false

# 强制辅助权重走 ModelScope（否则 need_proxy() 会去 TCP 探测 huggingface.co）
export USE_MODELSCOPE=true

# 遥测
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$UV_CACHE_DIR" "$MODELSCOPE_CACHE" "$HF_HOME" "$TORCH_HOME" \
         "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" \
         "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR"
