"""Figure 7, appendix: the drift of figure 7 with, in addition, the unconstrained run with BorderZero projection
(`none_border0`), labelled as in `common.NAMES`. Same computation and layout as `fig7_aposteriori_drift.py`; saved as
`fig7_appendix_drift_{field}.pdf` in the `fig7` subfolder of FIGURES."""
import fig7_aposteriori_drift as fig7
from common import NAMES

METHODS = fig7.METHODS | {'none_border0': NAMES['none_border0']}
COLORS = fig7.COLORS + ['C3']


def compute():
    """`fig7_aposteriori_drift.compute` of METHODS."""
    return fig7.compute(METHODS)


def plot(computed):
    """`fig7_aposteriori_drift.plot` of `computed` with METHODS and COLORS: {field: figure}."""
    return fig7.plot(computed, METHODS, COLORS)


def main():
    """`fig7_aposteriori_drift.main` of METHODS and COLORS: plot and save the figures."""
    fig7.main(METHODS, COLORS, 'fig7_appendix_drift')


if __name__ == '__main__':
    main()
