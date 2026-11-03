"""Figure 8, appendix: the drift maps of figure 8 with, in addition, the unconstrained run with BorderZero projection
(`none_border0`).

Same computation and layout as `fig8_drift_spatial.py` (experiment EXPERIMENT, sweep PREFIX): for each method of
METHODS, one map of the thickness-weighted column RMSE of T to day 0, averaged over the first FIRST_DAYS daily outputs
and divided by its domain mean, one shared logarithmic colour scale. Its methods are those of figure 8, plus
`none_border0` titled as in `common.NAMES`, as an extra panel on the right. Saved as `fig8_appendix_drift_spatial.pdf`
in FIGURES. A method whose run is not collected yet is skipped.
"""
from fig8_drift_spatial import METHODS as FIG8_METHODS, main
from common import NAMES

METHODS = FIG8_METHODS | {'none_border0': NAMES['none_border0']}

if __name__ == '__main__':
    main(METHODS, 'fig8_appendix_drift_spatial')
