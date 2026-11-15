#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist=gpu42
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/browser-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-demo-g3-catalog-20261115/logs/browser-%j.err
set -euo pipefail
cd /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review
export PYTHONPATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/code
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/persist_data/home/guisongxia/.local/node/bin:$PATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
mkdir -p .runtime/demo-g3-review/browser-tmp
export TMPDIR="$PWD/.runtime/demo-g3-review/browser-tmp"
export PLAYWRIGHT_BROWSERS_PATH=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/browsers
export PLAYWRIGHT_MODULE=/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/browser-tools/node_modules/playwright
(
cd .runtime/demo-g3-review/browser-deps/packages
apt-get download libwayland-server0
for package in libwayland-server0*.deb; do dpkg-deb -x "$package" ../root; done
)
export LD_LIBRARY_PATH="$PWD/.runtime/demo-g3-review/browser-deps/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
node tools/agent_baseline/check_demo_g3_browser.cjs results/agent_baseline/demo_g3 /guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/cache/browser_audit.json
