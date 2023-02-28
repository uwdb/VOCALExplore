import numpy as np

def ensure_list(x):
    """Ensures that x is a list.

    Args:
        x: A value.

    Returns:
        A list.
    """
    if isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return x.tolist()

    return [x]

def ensure_str(x, sep='+'):
    """Ensures that x is a string.
    If x is a list, it will be joined with sep.

    Args:
        x: A value.
        sep: join separator.

    Returns:
        A string.
    """
    if isinstance(x, str):
        return x

    def check_collision(i):
        assert sep not in i, f'Cannot have {sep} in {i}'
    [check_collision(i) for i in x]

    return sep.join(x)
