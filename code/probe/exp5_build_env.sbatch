#!/usr/bin/env bash
# Build conda_envs/exp5-env for experiment 5 (AnyChange via torchange, RemoteCLIP via open_clip).
# A new, independent environment; omniovcd-env and vlm-vs-onejev-env are not touched.
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
ENV=/guisongxia01/pangchao/wangyihan/conda_envs/exp5-env
CONDA=/persist_data/apps/miniconda3/bin/conda
WHEELS=/guisongxia01/pangchao/wangyihan/wheels/exp5
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TMPDIR=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/.tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "node=$(hostname -s)"
curl -s -o /dev/null -m 20 -w "mirror http %{http_code}\n" "$MIRROR/torch/"
if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main python=3.11
fi
PIP="$ENV/bin/python -m pip"
# PyPI torch 2.5.1 wheels bundle CUDA 12 runtimes; the tsinghua mirror serves them.
$PIP install -q -i "$MIRROR" torch==2.5.1 torchvision==0.20.1
$PIP install -q -i "$MIRROR" "$WHEELS"/torchange-0.0.4-py3-none-any.whl open_clip_torch scikit-image safetensors scipy pillow
"$ENV/bin/python" - <<'EOF'
import importlib.metadata as md
import torch, open_clip, torchange
from torchange.models.segment_any_change import AnyChange
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
for p in ("torchange", "open_clip_torch", "scikit-image", "torchvision"):
    print(p, md.version(p))
EOF
