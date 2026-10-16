# CA_0725 Box authorization and SimpleClick component scope

Date: 2026-07-25

## Runtime changes

- Staged point and box plans pass only their selected `region_id` as trusted
  metadata; the identifier is not added to the model-authored Action JSON.
- The Environment resolves the identifier against the live
  `verifier_region_proposals`. A point must equal the proposal component seed and a
  box must exactly equal the proposal `box_pixels`; mismatches fail before the
  segmentation backend is called.
- An authorized negative point extracts the white connected component containing the
  click from the proposal ROI. Only that component is sent as SimpleClick's
  `prev_mask`, and removal is limited to that component.
- Evidence records the proposal ROI size, localized session size, discarded proposal
  pixels, authorization source, composition mode, and proposal change ratio.

## Validation

Slurm targeted Job `46768` and full-regression Job `46769` completed successfully;
the full suite passed all `188` tests. Fixed-action CPU Job `46777` compared the same
three negative clicks on `test_20_15`:

| Region | Proposal ROI IoU | Clicked-component IoU |
| --- | ---: | ---: |
| r2 | 0.64345954 | 0.83669463 |
| r0 | 0.84913408 | 0.84846866 |
| r1 | 0.84691314 | 0.84876048 |
| Mean | 0.77983559 | 0.84464126 |

Initial IoU was `0.84638554`. Component-local input removed the severe r2
degradation and improved two of three candidates relative to proposal-ROI input. The
r0 candidate decreased slightly relative to proposal-ROI input but remained above the
initial IoU. The evidence supports retaining the change.

The full Qwen/Bailian post4 run is still pending. New GPU batch jobs, including a
minimal `nvidia-smi` job, were cancelled by Slurm with signal 53 before their scripts
started. This is a cluster GPU-job startup failure, not a code, network, or endpoint
probe failure.

Fixed-action evidence:

```text
outputs/CA_0725(4)-simpleclick-component-input-fixed-actions/comparison.json
```
