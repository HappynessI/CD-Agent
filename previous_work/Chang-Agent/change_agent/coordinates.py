"""Coordinate conversions at the public-protocol/Environment boundary."""

from __future__ import annotations

from collections.abc import Sequence


PROTOCOL_COORDINATE_MAX = 1000
PROTOCOL_COORDINATE_SPACE = "normalized_0_1000"


def normalized_point_to_pixel(
    point: Sequence[float],
    image_size: tuple[int, int],
    *,
    coordinate_max: float = PROTOCOL_COORDINATE_MAX,
) -> tuple[int, int]:
    """Convert a public normalized XY point to an internal pixel XY point."""

    if len(point) != 2:
        raise ValueError("point must contain two coordinates")
    width, height = _validate_image_size(image_size)
    x, y = (float(value) for value in point)
    if not (0 <= x <= coordinate_max and 0 <= y <= coordinate_max):
        raise ValueError(f"coordinates must be normalized to [0, {coordinate_max:g}]")
    return (
        round(x / coordinate_max * (width - 1)),
        round(y / coordinate_max * (height - 1)),
    )


def pixel_point_to_normalized(
    point: Sequence[int], image_size: tuple[int, int]
) -> tuple[int, int]:
    """Convert an internal pixel XY point to the public normalized space."""

    if len(point) != 2:
        raise ValueError("point must contain two coordinates")
    width, height = _validate_image_size(image_size)
    x, y = (int(value) for value in point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("pixel coordinate is outside the image")
    return _scale_pixel(x, width), _scale_pixel(y, height)


def pixel_box_to_normalized(
    box: Sequence[int], image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert an inclusive internal pixel XYXY box to public normalized XYXY."""

    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    width, height = _validate_image_size(image_size)
    x1, y1, x2, y2 = (int(value) for value in box)
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise ValueError("pixel box must be ordered and inside the image")
    left = pixel_point_to_normalized((x1, y1), (width, height))
    right = pixel_point_to_normalized((x2, y2), (width, height))
    return left[0], left[1], right[0], right[1]


def validate_normalized_box(box: Sequence[int]) -> tuple[int, int, int, int]:
    """Validate a public normalized XYXY region without converting it to pixels."""

    if len(box) != 4:
        raise ValueError("normalized box must contain four coordinates")
    x1, y1, x2, y2 = (int(value) for value in box)
    if not (
        0 <= x1 <= x2 <= PROTOCOL_COORDINATE_MAX
        and 0 <= y1 <= y2 <= PROTOCOL_COORDINATE_MAX
    ):
        raise ValueError("normalized box must be ordered inside [0, 1000]")
    return x1, y1, x2, y2


def _scale_pixel(value: int, extent: int) -> int:
    if extent == 1:
        return 0
    return round(value / (extent - 1) * PROTOCOL_COORDINATE_MAX)


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must be (width, height)")
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height
