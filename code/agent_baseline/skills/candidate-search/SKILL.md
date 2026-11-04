---
name: candidate-search
when: When the images show changed objects that your current answer misses.
basis: General method; coarse segmentation tends to miss dense small objects. Written 2026-11-04.
---
# Candidate search

Search only where you can see a likely miss. Most added regions in development runs were wrong, so look first.

1. Find the area with `view`: changed objects with no contour on them, often small objects packed closely.
2. Call `segment_text` with `tile` set to 256 and `region` set to that area, at both dates. Smaller tiles zoom in; they find smaller objects and also more false ones.
3. Call `match_instances` on the two sets. Single-date instances say nothing about change.
4. Call `to_mask` with `set_current=false`, and look at the candidate before you use it.
5. Use `add_region` for the components you believe, not the whole candidate.

The verifier checks the added components when you submit. If the area holds no visible miss, stop searching.
