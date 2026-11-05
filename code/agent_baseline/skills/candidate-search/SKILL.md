---
name: candidate-search
when: When the images show changed objects that your current answer misses.
basis: General method; coarse segmentation tends to miss dense small objects. Written 2026-11-04.
---
# Candidate search

Search only where you can see a likely miss. A region that looks new may exist at one date only, or be a false detection, so look first.

1. Find the area with `view`: changed objects with no contour on them, often small objects packed closely.
2. For one object, draw a tight box and call `segment_prompt` at the date it exists. For many objects, call `segment_text` at both dates with a lower `score_thr`, then `match_instances`. Single-date instances say nothing about change.
3. Call `to_mask` with `set_current=false`, and look at the candidate before you use it.
4. Use `add_region` for the components you believe, not the whole candidate.

The verifier checks the added components when you submit. If the area holds no visible miss, stop searching.
