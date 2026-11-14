#!/usr/bin/env python3
"""Write the paired job using the user-confirmed compute node; does not submit."""
import json
from pathlib import Path
import argparse
import shlex

repo=Path(__file__).resolve().parents[2]
runtime=repo/'.runtime/gate6-actual-20261113'
p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);args=p.parse_args()
exp=args.experiment.resolve()
# User confirmed gpu42 + CSV workspace endpoint: HTTP 200, 182 models.
# This is user-provided evidence, not a measurement by the cancelled probe.
node='gpu42'
manifest=args.manifest.resolve()
tasks=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
if len(tasks)!=10:raise SystemExit('Need exactly 10 paired development tasks')
snapshot=runtime/'code/agent_baseline';online=runtime/'online';output=repo/'results/agent_baseline/gate6_actual_20261113/paired'
common=['--tasks',str(manifest),'--split','val','--concurrency','2','--prompt-version','v4','--seed-offset','1',
    '--batch','gate6_actual','--max-cost-cny','100','--planner-model','qwen3.7-flash','--ask-backend','api','--ask-model','qwen3.7-flash',
    '--ask-correspondence','--pixel-thr','0.85','--status','--verify-stage','--gate5','--tools-t13','--tool-hints','a',
    '--vision-config',str(snapshot/'vision_t13.example.json')]
command=['bash',str(snapshot/'run_agent.sbatch'),*common,'--config','gate6_off',':::',*common,'--config','gate6_on','--gate6']
script=f'''#!/usr/bin/env bash
#SBATCH --partition=gre
#SBATCH --nodelist={node}
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={exp}/logs/paired-%j.out
#SBATCH --error={exp}/logs/paired-%j.err
set -euo pipefail
export AGENT_RUN={shlex.quote(str(online))}
export AGENT_CODE={shlex.quote(str(snapshot))}
export OVCD_CODE={shlex.quote(str(repo/'code/ovcd_v0'))}
export AGENT_STOP_FILE={shlex.quote(str(online/'STOP.json'))}
export BAILIAN_CSV={shlex.quote(str(repo.parent/'默认业务空间-apiKey-6214720.csv'))}
export USE_ASK_WORKER=0 USE_LOCAL_WORKERS=1
export PYTHONPATH={shlex.quote(str(runtime/'code'))}
report_results() {{
    status=$?
    trap - EXIT
    set +e
    PYTHONPATH={shlex.quote(str(repo/'code'))} /usr/bin/python3 -B -m agent_baseline.analysis.gate6_pair_report --run {shlex.quote(str(online))} --manifest {shlex.quote(str(manifest))} --output {shlex.quote(str(output))}
    reported=$?
    if [[ "$status" == 0 ]]; then status=$reported; fi
    exit "$status"
}}
trap report_results EXIT
{shlex.join(command)}
'''
(exp/'jobs/03-paired.sbatch').write_text(script)
(exp/'paired_settings.json').write_text(json.dumps(dict(node=node,network_mode='direct',tasks=str(manifest),
    common_arguments=common,arms=['gate6_off','gate6_on'],only_arm_difference='--gate6',estimated_cost_cny=5),indent=2)+'\n')
print('Prepared same-node paired job: '+str(exp/'jobs/03-paired.sbatch'))
