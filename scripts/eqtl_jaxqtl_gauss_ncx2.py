#!/usr/bin/env python3
"""Run the jaxQTL 0.3.1 CLI unchanged except that Beta-permutation calibration uses the
non-central chi-square reference (use_tdist=False, the reference the CLI uses for count models)
instead of the Student-t reference the CLI selects for the Gaussian family. Used only for the
Gaussian permutation run, because the Student-t reference fit stops with "Nonlinear solve
diverged" on this data (see README)."""
import sys

import jaxqtl.cli as cli

_BetaPermutation = cli.BetaPermutation


def _beta_permutation_ncx2(*args, **kwargs):
    kwargs["use_tdist"] = False
    return _BetaPermutation(*args, **kwargs)


cli.BetaPermutation = _beta_permutation_ncx2
sys.exit(cli.run_cli())
