"""Figure 8, appendix: the drift maps of figure 8 with, in addition, the unconstrained run with BorderZero projection
(`none_border0`), titled as in `common.NAMES`, as an extra panel on the right. Same computation and layout as
`fig8_drift_spatial.py`; saved as `fig8_appendix_drift_spatial.pdf` in FIGURES."""
import fig8_drift_spatial as fig8
from common import NAMES

METHODS = fig8.METHODS | {'none_border0': NAMES['none_border0']}


def compute():
    """`fig8_drift_spatial.compute` of METHODS."""
    return fig8.compute(METHODS)


def plot(computed):
    """`fig8_drift_spatial.plot` of `computed` with METHODS: the figure."""
    return fig8.plot(computed, METHODS)


def main():
    """`fig8_drift_spatial.main` of METHODS: plot and save the figure."""
    fig8.main(METHODS, 'fig8_appendix_drift_spatial')


if __name__ == '__main__':
    main()
