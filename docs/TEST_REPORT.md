# Local Verification (2026-11-04)

Executed on the local CPU environment, not the manuscript's RTX 4090 workstation.

- Four unit tests passed: input/domain/field bounds; PyTorch vs NumPy residuals;
  portable weight export; truncated proposals and bundled dependency resolution.
- Three-step CPU training with batch 16 completed and wrote checkpoints, all
  component losses, memory samples and a summary. This is not a convergence test.
- A ten-point prepared-batch CPU benchmark completed with three timed repeats
  and zero conventional solver-status failures. It is not the manuscript's
  1000-point benchmark, and its timing is not used to claim the historical 5.25x.
- Eight-point local-matter validation completed with finite field, energy and
  pressure errors. It is not a reproduction of Table III.
- A two-proposal nuclear-only process-pool probe completed, retaining both
  rejection records (minima outside the saturation bracket). This verifies
  failure recording, not a nonzero acceptance rate.
- The optimized reference nuclear calculation completed (n0 about 0.14734,
  Esym about 28.569 MeV), consistent at the numerical precision of this screening
  configuration. The precise manuscript table uses its own derivative settings.
- The original-reference 400-density EOS completed the 100-node stellar scan,
  maximum refinement and targeted mass solves with no failures: Mmax about
  2.093917 solar masses and R1.4 about 13.009621 km.
- The R2 input loader/KDE/empty-bank reporting path completed using the included
  actual observation files. No mock data are substituted.
- A one-curve selection fixture using the computed original-reference stellar
  sequence exercised the full MR/GW profiling path; it passed the separate 90%
  and 95% checks. The fixture is not counted as a sampled R1 EOS.
- Python source compilation completed for scripts, tests and bundled toolkit.

Not performed: 200000-step retraining, 432000-proposal sampling, new full ensemble
figures, full beta-equilibrium PINN-vs-exact validation, GPU resource profiling,
or independent review of upstream data/code redistribution permissions.

The historical measurements in reference_results remain explicitly historical.
