"""Settings transcribed from the author-approved manuscript (2026-11-04)."""
ORIGINAL = dict(g_sigma=-10.2143, g_omega=13.4353, g_rho=7.2687,
                g_a0=-6.7, g2=1596.3108, g3=5.8809, c3=172.1484,
                c4=204.0879, b6=180.3132, m_sigma=491.5)
LOSS_REDUCTION = "sum_of_batch_means"
PROPOSAL_RELATIVE_SIGMA = 0.40
NUCLEAR_INTERVALS = dict(rho0=(.11, .21), Ebind=(-17., -15.),
                         K=(200., 260.), J=(27.7, 34.1), L=(35., 70.))
