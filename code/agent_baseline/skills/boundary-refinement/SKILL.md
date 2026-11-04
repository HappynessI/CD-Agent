---
name: boundary-refinement
when: When one component's outline is visibly wrong, for example it spills across a neighbour or a road.
basis: General method; development runs showed that replacing outlines with box prompts often made them worse. Written 2026-11-04.
---
# Boundary refinement

Most outlines are good enough. Refine only a component whose error you can see, and only when the error is large relative to the component.

1. Zoom in with `view` on the component at both dates.
2. Prefer the smallest edit. `morph` with `open` and a small radius removes thin spills, but it acts on the whole mask, so pass `set_current=false` and compare the result.
3. Use `refine_region` with a tight box only when the object is clear. Read `sam_quality` as segmentation quality, not as a sign that the change is real.
4. Compare the new outline with the old one in a view. Keep the old one if the new one is not clearly better.

An edited outline has no verifier answer. It goes to the verifier when you submit.
