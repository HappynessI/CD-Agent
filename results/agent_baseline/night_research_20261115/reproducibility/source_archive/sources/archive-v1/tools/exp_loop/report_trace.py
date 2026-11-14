#!/usr/bin/env python3
"""Summarize existing trace diagnostics; executed inside the experiment job."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--phase',required=True)
a=p.parse_args()
data=json.loads((a.checkpoint/'summary.json').read_text())
count=sum(g['n'] for g in data['groups'])
a.output.mkdir(parents=True,exist_ok=True)
(a.output/f'{a.phase}_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
lines=['# 轨迹行为诊断阶段结果', '',f'阶段：{a.phase}；会话：{count}；读取失败：{len(data["read_failures"])}；API费用：0元。',
       '本报告是事件统计，F1差异不是因果归因；语义近似重复、核验器空间推翻及净收益尚未覆盖。',
       '完整逐会话事件与续跑账本：'+str(a.checkpoint/'sessions.jsonl'),'',
       '| 配置 | 任务 | 会话 | 正常提交轮数中位数 |','|---|---|---:|---:|']
for g in data['groups']:
    lines.append(f'| {g["config"]} | {g["task"]} | {g["n"]} | {g["normal_round_median"]} |')
(a.output/f'{a.phase}_report.md').write_text('\n'.join(lines)+'\n')
(a.output/f'{a.phase}_metric.json').write_text(json.dumps(dict(primary=count,primary_name='sessions_analyzed',
    cost_cny=0,read_failures=len(data['read_failures']),phase=a.phase))+'\n')
print(f'phase={a.phase} sessions={count} read_failures={len(data["read_failures"])}')
if data['read_failures']:
    raise SystemExit('Some traces could not be read; partial evidence saved, do not call this complete')
