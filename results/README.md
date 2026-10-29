# Recorded experiments

`world1-evaluation.csv` contains the five CPU argmax evaluations of each selected level model. Each episode uses a newly created environment, seeds 0 through 4, and the standard map and starting point. The released weight manifest identifies the corresponding model tensors.

`ablation-seed42.csv` summarizes the completed four-mode reward comparison on World 1-1. All modes use 5,242,880 sampled decisions and one training seed. The endpoint columns describe one CPU argmax episode at the exact final budget. Periodic-clear columns describe a sequence of different model snapshots, not repeated trials of a fixed model.

`learning-curves/` contains the original metrics and evaluation CSV files for the selected PPO runs, the reward comparison, and the behavior-cloning experiment. Numerical values have not been resampled or edited. `data-manifest.json` records file hashes.

Training-return fields are rolling means over at most 100 completed episodes in the current process segment. A resumed run resets its environments and rolling window; use `segment_id` to avoid joining curves across that boundary. Incomplete evaluations have `complete=False` and must be excluded from completed-trial counts.

The reported reward subset was reduced to seed 42 after its results were known. Native and progress_terminal each resumed once; progress_score and full did not. Equal sampled decisions did not produce identical optimizer-update counts. These data support a comparison of this recorded seed, not a general ranking across seeds or levels.
