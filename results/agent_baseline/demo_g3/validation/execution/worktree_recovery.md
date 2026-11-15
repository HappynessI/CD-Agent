# Worktree recovery

The initial full checkout reached 73,431 files and failed while resetting the index. Git removed the worktree files in its failed-creation cleanup. The experiment runtime, inventory and generated page cache survived.

The branch was reopened with a no-checkout worktree and its metadata was repaired at the original working path. The base code, documentation and original G3 pages were copied from the primary worktree. Fifteen edit commands from this task were reapplied in order. The original selection-list format was then checked. The final CPU job repeats inventory, generation, static audit, regression tests and offline browser checks from the recovered source.
