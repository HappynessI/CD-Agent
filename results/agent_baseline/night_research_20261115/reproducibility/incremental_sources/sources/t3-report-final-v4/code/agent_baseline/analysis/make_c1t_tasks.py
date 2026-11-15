"""T1 IDs {1,2} -> T2 ID 5; opposite is T1 5 -> T2 {1,2}.

1=low vegetation; 2=N.V.G surface (non-vegetated ground surface);
5=building; 0=unlabelled unchanged. Both dates must be annotated.
"""
import numpy as np
from .second_labels import building_truth


def conversion_truth(a, b):
    building_truth(a, b)
    return dict(new=np.isin(a, [1, 2]) & (b == 5),
                demolished=(a == 5) & np.isin(b, [1, 2]))


if __name__ == '__main__':
    from .make_t12_tasks import main
    main('c1t')
