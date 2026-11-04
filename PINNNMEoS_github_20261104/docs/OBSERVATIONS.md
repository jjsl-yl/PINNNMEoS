# Observational Inputs

The processed MR grids and the GW table from the local analysis are included.
They are not automatically downloaded or relicensed by this bundle. Use the
published releases and retain their citations and original metadata. The release
maintainer must check upstream data redistribution terms before public hosting.

- J0740: Salmi et al. (2024), https://zenodo.org/records/10519473
- J0437: Choudhury et al. (2024), https://arxiv.org/abs/2407.06789
- GW170817: low-spin PhenomPNRT, release LIGO-P1800061-v11,
  https://dcc.ligo.org/LIGO-P1800061/public

Required files under the directory passed to `--observations`:

```text
j0740_mr_posterior_grid.npz
j0437_mr_posterior_grid.npz
gw170817/low_spin_PhenomPNRT_posterior_samples.dat.gz
```

Each MR NPZ must contain `mass_axis_Msun`, `radius_axis_km`, and `density`, with
density shape (number of mass nodes, number of radius nodes). Axes must be
increasing. These are processed density grids, not names guaranteed to occur in
the original release. The bundle does not invent or substitute mock observations.
Use the exact processed grids used for the manuscript, or construct grids from
the published joint samples and document the estimator and bandwidth. For a
public exact-data release, preserve the supplied checksums and add the original
grid-processing provenance. That provenance is not fully encoded in the NPZ
files; the original processing cannot be reconstructed from the grid alone.

The GW table must retain named columns `m1_detector_frame_Msun`,
`m2_detector_frame_Msun`, `lambda1`, and `lambda2`. Detector masses are divided
by 1+0.0099. KDE uses standardized coordinates, bandwidth 0.25 and boundary
reflection; 7 chirp-mass x 80 mass-ratio profiling nodes are used. Thresholds
are empirical quantiles of KDE values at the sample locations. MR curves use
1000 supported mass nodes. These approximations are not a calibrated combined
confidence construction. J0030 and reconstructed GW MR contours are not extra
R2 constraints.
