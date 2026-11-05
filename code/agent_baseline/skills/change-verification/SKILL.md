---
name: change-verification
when: When you read verifier answers, or a component's answer looks doubtful.
basis: General method for verifying change on image pairs; development runs showed the verifier errs in both directions. Written 2026-11-04.
---
# Change verification

A verifier answer is one reading of one outline. It says less than it appears to.

Read an answer in this order:
1. Check the answer against its own reason. If the reason describes the same object at both dates and the answer is yes, the answer is doubtful.
2. Look at the object at both dates. A matching outline, structure and position relative to fixed neighbours is strong evidence of the same object, whatever its colour, shadow or tone. Check for structural change before you conclude: an object can be rebuilt at the same place.
3. An outline that covers only part of an object, or spills over neighbours, gives a weak answer.

Look again with `view` where a wrong decision costs most:
- large components answered no, because removing them loses the most if the answer is wrong;
- components where `p_yes` is far from both 0 and 1, although a value near 0 or 1 can also be wrong;
- components where the answer and the images disagree.

Asking the same question again adds little.

The latest answer does not cancel an earlier one. If answers disagree, decide from the images and say so in the summary.
