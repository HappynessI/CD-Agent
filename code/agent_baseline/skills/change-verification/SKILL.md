---
name: change-verification
when: When you read verifier answers, or a component's answer looks doubtful.
basis: General method for verifying change on image pairs; development runs showed the verifier errs in both directions. Written 2026-11-04.
---
# Change verification

A verifier answer is one reading of one outline. It says less than it appears to.

Read an answer in this order:
1. Check the answer against its own reason. If the reason describes the same object at both dates and the answer is yes, the answer is doubtful.
2. Look at the object at both dates. The same roof outline at the same place relative to roads and neighbours is the same building, whatever its colour, shadow or tone.
3. An outline that covers only part of an object, or spills over neighbours, gives a weak answer.

Doubtful components deserve the thorough check `ask` with `level` set to `L2`. Spend it where a wrong decision costs most:
- large components answered no, because removing them loses the most if the answer is wrong;
- components where `p_yes` is far from both 0 and 1;
- components where the answer and the images disagree.

Do not spend it on small components with a clear answer. Asking again at the same level adds little.

The latest answer does not cancel an earlier one. If answers disagree, decide from the images and say so in the summary.
