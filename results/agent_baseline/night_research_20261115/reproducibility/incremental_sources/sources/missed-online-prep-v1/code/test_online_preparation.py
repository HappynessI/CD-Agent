import unittest,json,os
from pathlib import Path
modules=["agent_baseline.tests.test_frozen_ranked_candidates","agent_baseline.tests.test_find_missed","agent_baseline.tests.test_missed_online_runner"]
suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
r=unittest.TextTestRunner(verbosity=2).run(suite)
p=Path('/guisongxia01/pangchao/wangyihan/wyh/CD-Agent/results/agent_baseline/night_missed_20261114/online_preparation_tests');p.mkdir(parents=True,exist_ok=True)
(p/"result.json").write_text(json.dumps(dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),successful=r.wasSuccessful(),slurm_job_id=os.environ["SLURM_JOB_ID"],agent_sessions=0,scope="Provider/runner preparation tests only; no online promotion or inference"),indent=2)+"\n")
raise SystemExit(0 if r.wasSuccessful() else 1)
