---
name: mask-correction
when: When you edit a mask, given or your own.
basis: General method; development runs lost most score by rebuilding masks and by adding unverified regions. Written 2026-11-04.
---
# Mask correction

Correct component by component. Keep the handle that holds your complete answer.

Removing:
- Use `remove_region`, which keeps the IDs of the other components.
- Remove a component when the verifier answers no and the images do not contradict that answer. Check a large component before you remove it.
- A large removed area is not a sign of a better result.

Adding:
- A view of one date shows what exists, not what changed. Compare both dates.
- Added regions go to the verifier when you submit. Read that answer before you keep them.
- Prefer adding one object at a time with `add_region`. Use a tight box if you use `segment_prompt`.

Do not replace a complete answer with a rebuilt mask unless each dropped region has been checked. If an edit went wrong, go back to the earlier handle; handles stay available.
