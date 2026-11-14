# R1 direct-four early input review

Main Agent inspected the four actual request PNGs for `tune/tasks/train_1136_5to1/verification/47` before reading its response. This is a targeted mechanism check, not an independent semantic truth annotation.

Example T1 shows a blue roof inside the thin red support contour, and T2 shows vegetation at the corresponding location. Candidate T1/T2 are the same spatial crop; the red support is a narrow strip at the upper edge, while much of the visible blue/gray roofing below it is outside the candidate. The original RGB is preserved and the Example/Candidate roles and time order are correct.

The response rejects the candidate, describing the example as impervious→vegetation and candidate as impervious→impervious. It now describes the visible example transition rather than treating the unchanged contour shape as a persistent building. However, the narrow candidate support leaves substantial contextual distraction, and the task has only 3 remaining positive GT pixels. Neither this response nor this case establishes a retrieval improvement or isolates the cause of the earlier Spotlight error.

Candidate/exemplar overlap is intentionally unchanged from the fixed R1 proposal pool. It must be analyzed when interpreting direct comparison results; scoring alone subtracts the exemplar. The first tune task `train_1065_2to5` has no remaining positive GT, so visually plausible repeated-exemplar matches are not evidence of successful retrieval.
