"""Reproduce the PyDESeq2 public-helper reference used in test_bulk.cpp."""
import importlib.metadata
import json
import numpy as np
import pandas as pd
from pydeseq2.utils import fit_rough_dispersions, fit_moments_dispersions

size_factors = np.array([.01, .02, 1, 1, 1, 1, 1, 1])
counts = np.array([0, 0, 5, 20, 10, 40, 5, 20])[:, None]
design = pd.DataFrame({'Intercept': np.ones(8), 'covariate': np.arange(8)})
normalized = counts / size_factors[:, None]
rough = fit_rough_dispersions(normalized, design)
moments = fit_moments_dispersions(normalized, size_factors)
print(json.dumps({'reference': 'PyDESeq2 ' + importlib.metadata.version('pydeseq2'),
                  'counts': counts[:, 0].tolist(), 'size_factors': size_factors.tolist(),
                  'rough': rough.tolist(), 'moments': moments.tolist(),
                  'clipped_minimum': np.maximum(np.minimum(rough, moments), 1e-8).tolist()}, indent=2))
