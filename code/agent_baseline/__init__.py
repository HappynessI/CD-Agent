"""BF agent tools. Install beside ovcd_v0, or set OVCD_CODE for staging."""
import os
import sys
from pathlib import Path

_sibling = Path(__file__).resolve().parent.parent / 'ovcd_v0'
_legacy = Path(os.environ.get('OVCD_CODE', str(_sibling if _sibling.is_dir() else
    Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/code/ovcd_v0'))))
# The reused modules themselves use script-style imports (common, prompts).
if str(_legacy) not in sys.path:
    sys.path.append(str(_legacy))
