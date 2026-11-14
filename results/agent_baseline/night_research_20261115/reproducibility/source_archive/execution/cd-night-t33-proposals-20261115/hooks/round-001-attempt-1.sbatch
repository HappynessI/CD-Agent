#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:05:00
#SBATCH --output=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t33-proposals-20261115/hooks/hook-%j.out
#SBATCH --error=/guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t33-proposals-20261115/hooks/hook-%j.err
set -euo pipefail
/usr/bin/python3 /guisongxia01/pangchao/wangyihan/wyh/CD-Agent/tools/exp_loop/loop.py advance /guisongxia01/pangchao/wangyihan/wyh/exp-loop/experiments/cd-night-t33-proposals-20261115 186472
