import json
from pathlib import Path

CACHE = Path("/guisongxia01/pangchao/wangyihan/wyh/cd-agent-worktrees/demo-g3-review/.runtime/demo-g3-review/cache")


def read_if_present(name):
    path = CACHE / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


static = read_if_present("page_audit.json")
browser = read_if_present("browser_audit.json")
passed = (
    static.get("total_pages", 0) > 0
    and static.get("legacy_selection_preserved") is True
    and static.get("historical_replays_verified") == 9
    and browser.get("status") == "passed"
)
print(json.dumps({
    "stage": "demo_g3_catalog",
    "cost_cny": 0,
    "validation_exists": bool(static and browser),
    "validation_passed": passed,
    "pages": static.get("total_pages"),
    "embedded_images": static.get("embedded_images"),
    "local_links_checked": static.get("local_links_checked"),
    "historical_replays_verified": static.get("historical_replays_verified"),
    "legacy_selection_preserved": static.get("legacy_selection_preserved"),
    "offline_browser_status": browser.get("status"),
}))
