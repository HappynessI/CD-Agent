#!/usr/bin/env python3
"""Install stable workspace entrypoints; leave experiments, state and cron unchanged."""
from pathlib import Path
import shutil
import shlex

repo=Path(__file__).resolve().parents[2]
source=repo/'tools/exp_loop'
destination=repo.parent/'exp-loop'
backup=repo/'.runtime/exp-loop-before-20261113'
backup.mkdir(parents=True,exist_ok=True)
for name in ('loop.py','watchdog.sh','README.md'):
    old=destination/name
    if old.exists() and not (backup/name).exists(): shutil.copy2(old,backup/name)
(destination/'loop.py').write_text('#!/usr/bin/env python3\nimport runpy\nrunpy.run_path('+repr(str(source/'loop.py'))+', run_name="__main__")\n')
(destination/'watchdog.sh').write_text('#!/usr/bin/env bash\nexport EXP_LOOP_ROOT='+shlex.quote(str(destination))+'\nexec bash '+shlex.quote(str(source/'watchdog.sh'))+'\n')
for name in ('loop.py','watchdog.sh'): (destination/name).chmod(0o755)
(destination/'README.md').write_text('技术说明统一维护于 [CD-Agent README_ai §10.17](../CD-Agent/doc/agent_baseline/README_ai.md#exp-loop)。\n\n版本化实现位于 CD-Agent/tools/exp_loop；本目录只保留入口、实验状态和日志。\n')
print('Installed entrypoints; preserved experiments and cron. Backup:',backup)
