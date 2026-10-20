#!/usr/bin/env bash
# Build conda_envs/exp6-v2-env for experiment 6 V2 (Qwen3.5-4B LoRA). vlm-vs-onejev-env is a
# venv layered on omniovcd-env; the new venv layers on the same base and reuses vlm's
# site-packages read-only through a .pth file, so only peft lands in the new env.
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
set -euo pipefail
SRC=/guisongxia01/pangchao/wangyihan/conda_envs/vlm-vs-onejev-env
ENV=/guisongxia01/pangchao/wangyihan/conda_envs/exp6-v2-env
BASE=/guisongxia01/pangchao/wangyihan/conda_envs/omniovcd-env/bin/python3.12
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export TMPDIR=/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/CD_1016_probe/.tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
echo "node=$(hostname -s)"
if [ ! -x "$ENV/bin/python" ]; then
  "$BASE" -m venv --system-site-packages "$ENV"
fi
echo "$SRC/lib/python3.12/site-packages" > "$ENV/lib/python3.12/site-packages/zz_vlm_vs_onejev.pth"
"$ENV/bin/python" -m pip install -q -i "$MIRROR" peft
"$ENV/bin/python" - <<'PY'
import importlib.metadata as md
for p in ("torch", "transformers", "peft", "accelerate"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "missing")
PY
du -sh "$ENV"
