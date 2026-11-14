"""Public feedback geometry. No labels, error IDs or error masks enter this module."""
import numpy as np

CELLS = ('upper left', 'top', 'upper right', 'left', 'center', 'right',
         'lower left', 'bottom', 'lower right')


def cell_at(point, shape):
    """Pixel (x,y) belongs to floor(3*x/width), floor(3*y/height).

    Equivalently each cell has exclusive upper bounds ceil((i+1)*size/3).
    A pixel exactly on a third boundary belongs to the right/bottom cell.
    """
    x, y = point
    h, w = shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Feedback point must lie inside the image')
    return CELLS[int(3 * y // h) * 3 + int(3 * x // w)]


def cell_mask(where, shape):
    row, col = divmod(CELLS.index(where), 3)
    h, w = shape
    out = np.zeros(shape, bool)
    out[(row*h+2)//3:((row+1)*h+2)//3,
        (col*w+2)//3:((col+1)*w+2)//3] = True
    return out


def public_feedback(row):
    """Whitelist only information rendered in the task instruction, even for empty feedback."""
    if row['task_type'] == 'T5.2':
        return dict(feedback_points=[tuple(e['point']) for e in row['meta']['feedback']])
    if row['task_type'] == 'T5.3':
        return dict(feedback_cells=[e['where'] for e in row['meta']['feedback']])
    return {}
