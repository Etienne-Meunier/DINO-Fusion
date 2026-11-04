"""Side-by-side comparison of the main paper figures with the increasing-density constraint at eta = 10 and eta = 30.

Each script of SCRIPTS is run once per run of RUNS, in its own process, with PAPER_DENSITY_RUN set to that run and
PAPER_FIGURES to `<FIGURES>/eta_compare/<label>` (see `common.py`), so the paper figures in FIGURES are left untouched.
FIGURES is the PAPER_FIGURES of the caller when set, else the `figures` folder next to this file, as in `common.py`.
The pages of FIGURE_FILES are then placed side by side, eta = 10 on the left and eta = 30 on the right (vector, through
PyMuPDF), one figure per page, in `<FIGURES>/eta_compare/eta10_vs_eta30.pdf`. If any script failed, nothing is assembled
and the script exits with an error, so the comparison never mixes fresh and stale renders.

Usage, from this folder: `python compare_eta.py` to render and assemble, `python compare_eta.py --assemble` to only
assemble the figures already rendered.
"""
import os
import subprocess
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.environ.get('PAPER_FIGURES', f'{HERE}/figures')
OUT = f'{FIGURES}/eta_compare'
RUNS = {'eta10': ('negative_grad_density_huber_border0', 'eta = 10'),
        'eta30': ('negative_grad_density_huber_border0_30', 'eta = 30')}
SCRIPTS = ['fig1_density_profile.py', 'fig2_state.py', 'fig5_aposteriori.py', 'fig6_amoc.py',
           'fig7_aposteriori_drift.py', 'fig8_drift_spatial.py']
FIGURE_FILES = ['fig1/fig1a.pdf', 'fig1/fig1b.pdf', 'fig2_state.pdf', 'fig5_aposteriori.pdf',
                'aposteriori/density_2m.pdf', 'aposteriori/amoc.pdf', 'fig7/fig7_drift_toce.pdf',
                'fig7/fig7_drift_soce.pdf', 'fig8_drift_spatial.pdf']
HEADER = 28  # height of the page header, in points
GAP = 20  # space between the two figures, in points


def render(label, constraint):
    """Run every script of SCRIPTS with `constraint` as the density run, saving into `OUT/label`; returns the failed scripts."""
    env = os.environ | {'PAPER_DENSITY_RUN': constraint, 'PAPER_FIGURES': f'{OUT}/{label}', 'MPLBACKEND': 'Agg'}
    failed = []
    for script in SCRIPTS:
        print(f'{label} : {script}', flush=True)
        if subprocess.run([sys.executable, script], cwd=HERE, env=env).returncode:
            failed.append(script)
    return failed


def assemble(path=f'{OUT}/eta10_vs_eta30.pdf'):
    """Write one page per file of FIGURE_FILES to `path`, the eta10 figure on the left and the eta30 one on the right,
    under a header with the file name and the eta of each side. A file missing on either side is skipped."""
    out = fitz.open()
    for name in FIGURE_FILES:
        sides = [f'{OUT}/{label}/{name}' for label in RUNS]
        if not all(os.path.exists(s) for s in sides):
            print(f'WARNING : {name} missing in {", ".join(s for s in sides if not os.path.exists(s))}, skipped')
            continue
        docs = [fitz.open(s) for s in sides]
        rects = [d[0].rect for d in docs]
        width = max(r.width for r in rects)
        height = max(r.height for r in rects)
        page = out.new_page(width=2 * width + GAP, height=height + HEADER)
        page.insert_text((6, 12), name, fontsize=10)
        for i, (doc, (_, title)) in enumerate(zip(docs, RUNS.values())):
            x0 = i * (width + GAP)
            page.insert_text((x0 + 6, HEADER - 4), title, fontsize=10)
            page.show_pdf_page(fitz.Rect(x0, HEADER, x0 + width, HEADER + height), doc, 0)
        for doc in docs:
            doc.close()
        print(f'{name} : page {out.page_count}')
    out.save(path)
    out.close()
    print(f'saved {path}')


if __name__ == '__main__':
    if '--assemble' not in sys.argv:
        failed = {label: render(label, constraint) for label, (constraint, _) in RUNS.items()}
        failed = {label: scripts for label, scripts in failed.items() if scripts}
        for label, scripts in failed.items():
            print(f'ERROR : {label} failed for {", ".join(scripts)}')
        if failed:
            sys.exit('not assembled: some renders failed')
    assemble()
