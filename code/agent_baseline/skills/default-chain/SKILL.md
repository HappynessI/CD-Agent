---
name: default-chain
when: At the start of a run, to choose the first steps.
basis: General change detection practice; the order was found useful in development runs. Written 2026-11-04.
---
# Default chain

Start from the plain chain unless the images give a reason not to.

With no given mask:
1. Call `segment_text` for the target class at T1 and at T2, in one reply.
2. Call `match_instances` on the two sets.
3. Call `to_mask` on the appeared and disappeared sets. This is your first complete answer.
4. Call `ask` with no arguments. Submitting also checks what is left, so you need not finish the list first.

With a given mask, the given mask is your first complete answer; go to step 4.

Then decide per component what stays, using `change-verification` and `mask-correction`.

Leave the chain only for a stated reason, such as objects the first answer clearly misses (`candidate-search`). A different parameter is not a better result by itself. Compare a new candidate with your current answer before you replace it.
