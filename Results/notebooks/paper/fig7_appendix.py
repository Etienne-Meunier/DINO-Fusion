"""Figure 7, appendix: the drift of figure 7 with, in addition, the unconstrained run with BorderZero projection
(`none_border0`).

Same computation and layout as `fig7_aposteriori_drift.py` (experiment EXPERIMENT, sweep PREFIX): for each method of
METHODS, the volume-weighted RMSE on each level of T and S to day 0, averaged over the last LAST_DAYS daily outputs and
smoothed along depth, one line per method, one panel per field. Its methods and colours are those of figure 7, plus
`none_border0` labelled as in `common.NAMES`. Saved as `fig7_appendix_drift_{field}.pdf` in the `fig7` subfolder of
FIGURES. A method whose run is not collected yet is skipped.
"""
from fig7_aposteriori_drift import COLORS as FIG7_COLORS, METHODS as FIG7_METHODS, main
from common import NAMES

METHODS = FIG7_METHODS | {'none_border0': NAMES['none_border0']}
COLORS = FIG7_COLORS + ['C3']

if __name__ == '__main__':
    main(METHODS, COLORS, 'fig7_appendix_drift')
