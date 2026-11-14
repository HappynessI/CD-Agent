# Fixed six-case verifier visual review

Viewed all six contact sheets before reading current five-arm responses; some cases were familiar from prior historical review, so not blind. No task/GT changed.

Main Agent visual inspection, not independent human truth

## second_appeared:train_1065:1

Red support covers a new light roof extension absent at T1; blue initial-mask overlay borders rather than occludes the candidate. All variants say yes, visually consistent. It is part of a larger new building, not necessarily an independent new object.

[Actual five-arm contact sheet](second_appeared__train_1065__1.png)

Decisions: V0=yes, V1=yes, V2=yes, V3=yes, V4=yes.

## second_appeared:train_1065:4

52px candidate is a small interior hole of the initial new-building mask, not a separate whole building. Original RGB context shows ground becoming roof; the blue overlay covers most of the surrounding roof. V0/V1/V3 say yes; V2/V4 reject as unchanged shape/color only. The fixed support shape is not evidence of persistent object identity; masking context may contribute, but this single comparison does not isolate cause.

[Actual five-arm contact sheet](second_appeared__train_1065__4.png)

Decisions: V0=yes, V1=yes, V2=no, V3=yes, V4=no.

## second_disappeared:train_1065:1

Small bright structured object at T1 is replaced by darker ground/vegetation at T2. Its fine semantic class is visually less certain than the large roofs. V0/V3 say demolished; V1/V2/V4 cite identical object shape/position and color-only change. The fixed highlighted region preserves shape regardless of object change, so these reasons require scrutiny; no GT is revised.

[Actual five-arm contact sheet](second_disappeared__train_1065__1.png)

Decisions: V0=yes, V1=no, V2=no, V3=yes, V4=no.

## second_disappeared:train_1065:2

Support is clipped by the image bottom; a bright/gray fragment remains visible in both dates. All variants reject. The narrow truncated support limits semantic certainty; do not attribute an error to rendering without checking the source image boundary.

[Actual five-arm contact sheet](second_disappeared__train_1065__2.png)

Decisions: V0=no, V1=no, V2=no, V3=no, V4=no.

## levir_any:train_106:1

Red support is a narrow linear feature near the upper-left road; new residential blocks are mostly outside it. V0/V3 call the candidate a new building complex, an observed region-attribution error. V1/V2/V4 reject, although V1 still calls the linear feature a building. Correct rejection alone does not validate its reasoning.

[Actual five-arm contact sheet](levir_any__train_106__1.png)

Decisions: V0=yes, V1=no, V2=no, V3=yes, V4=no.

## levir_any:train_106:3

Red support follows the shoreline/beach rather than adjacent houses; close/wide are effectively the whole image. All five variants reject a building change. V1 additionally says nearby buildings remain unchanged, which is unnecessary and not supported by the visible housing development. Prior structured-SoM failure is a different historical prompt arm, not evidence that this V3 failed here.

[Actual five-arm contact sheet](levir_any__train_106__3.png)

Decisions: V0=no, V1=no, V2=no, V3=no, V4=no.

The observations support separating full-object omission from holes/boundary repair before further prompt changes. A tune-only candidate-granularity audit is being prepared. This is a new hypothesis, not a measured gain or an online policy.
