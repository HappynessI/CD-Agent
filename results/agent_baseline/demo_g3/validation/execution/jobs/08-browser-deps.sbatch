#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/browser-deps-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/browser-deps-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/code
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p .runtime/demo-g3-review/browser-deps/packages
cd .runtime/demo-g3-review/browser-deps/packages
apt-get download libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxdamage1 libgbm1 libxkbcommon0 libasound2
for package in *.deb; do dpkg-deb -x "$package" ../root; done
