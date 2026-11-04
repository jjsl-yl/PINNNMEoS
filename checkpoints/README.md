No checkpoint trained with the new manuscript-aligned configuration is bundled.
`historical_validation_model.npz` is a portable export of the original validation
checkpoint. It retains the original optimized-parameter scaling and is provided
only for historical inference/benchmark checks, not as a newly trained model.
Run training, then export `latest.pt` using `scripts/export_checkpoint.py`.
Never replace the input scales of an old checkpoint and assume its predictions
are unchanged. The existing project checkpoint remains untouched outside this
release directory.
