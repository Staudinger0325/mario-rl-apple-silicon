# Training and experiments

## PPO

The agent receives four stacked 84×84 grayscale frames. An action is held for up to four emulator steps; the last two processed frames are max-pooled. Seven discrete actions cover no input, left, right, jump, run, and useful combinations. The shared network has four 32-channel convolution layers, a 512-unit fully connected layer, and separate seven-action policy and scalar value heads.

The World 1 PPO runs used eight CPU environments and 128 decisions per environment per rollout. A batch of 1,024 transitions was optimized in minibatches of 256 for at most four epochs on MPS. Adam used learning rate 0.00025 and epsilon 0.00001; gamma was 0.99, GAE lambda 0.95, clip range 0.2, value coefficient 0.5, entropy coefficient 0.01, and gradient norm limit 0.5. Value loss used Huber loss with beta 1. An epoch-average approximate KL above 0.03 stopped subsequent epochs. Advantages were normalized over the full rollout batch. Sampling used the policy distribution; evaluation used CPU argmax.

This example starts a new 1-1 run with the same main recipe. It will perform substantial computation and create checkpoints; change the budget or output path as needed.

```bash
.venv/bin/python -m mario_rl train --init scratch --world 1 --stage 1 \
  --steps 3376128 --num-envs 8 --rollout-steps 128 --minibatch-size 256 \
  --epochs 4 --lr 0.00025 --device mps --env-backend process \
  --reward-mode full --value-loss huber --seed 42 \
  --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 \
  --entropy-coef 0.01 --value-coef 0.5 --max-grad-norm 0.5 --target-kl 0.03 \
  --eval-every 32768 --eval-episodes 1 --eval-device cpu \
  --stop-after-clear-evals 3 --output runs/ppo-1-1
```

Use `python -m mario_rl train --help` in the virtual environment for all options. `--steps` counts new outer decisions, each of which can advance up to four emulator steps. `--resume` restores the model, optimizer, progress, and random-number state from a checkpoint; environments restart, so continuation is not a frame-exact replay. Training outcomes can vary with sampling and execution history. The published inference weights contain model tensors rather than optimizer state; a new scratch run will not reproduce the recorded curves bit for bit.

## Reward comparison

For each emulator step, let `n` be the environment reward divided by 10, `s` the score change divided by 400, and `t` +5 on a successful genuine termination, -5 on another genuine termination, and zero otherwise. The outer decision sums its constituent step rewards. The native reward already includes progress, time, and death signals.

The following 1-1 comparison used seed 42, the same initial network tensors and training recipe, and 5,242,880 sampled decisions per mode. Success did not stop training. Each final checkpoint cleared its single seed-0 greedy endpoint evaluation.

| Mode | Reward | First clear | First three consecutive clears | Periodic clears | Last 10 | Endpoint raw return |
|---|---|---:|---:|---:|---:|---:|
| native | n | 2,719,744 | 3,440,640 | 55/159 | 10/10 | 3,066 |
| terminal | n + t | 3,244,176 | 4,161,680 | 24/159 | 8/10 | 3,064 |
| score | n + s | 2,654,208 | 3,145,728 | 62/160 | 9/10 | 3,067 |
| full | n + s + t | 2,523,136 | 3,342,336 | 59/160 | 7/10 | 3,067 |

The CLI names for terminal and score are `progress_terminal` and `progress_score`. Periodic evaluations use successive training snapshots, not independent trials of one fixed model. Full reached its first clear earliest; score first reached three consecutive clears earliest; native had the best last-ten result. One seed does not establish a general ranking or estimate variability across seeds. Compare raw returns and clears across modes rather than their differently defined shaped returns.

The [evaluation results](../results/world1-evaluation.csv), [reward comparison](../results/ablation-seed42.csv), and [learning curves](../results/learning-curves/) are available as CSV files.

## 1-3: own search and behavior cloning

PPO attempts on 1-3 did not reach a successful frozen endpoint in the tested budgets. The project instead searched from an empty archive, replaying discovered prefixes from the start and extending them with random actions held for 1, 2, 4, or 8 steps. RAM position and region organized the search archive; the final policy receives only pixels. Four search runs used 928,993 interactions, including 859,429 for prefix replay.

The earliest verified seed-42 route supplied 439 observation-action pairs. A fresh random CNN learned them with cross-entropy, updating the shared trunk and policy head while freezing the value head. Training used Adam at 0.001, batch size 64, and stopped after five consecutive CPU training-set accuracy checks reached 100%; it took 113 epochs and 791 optimizer updates. The resulting policy cleared five independent fixed-start evaluations.

A subsequent 131,072-decision PPO probe used a fresh Adam optimizer, learning rate 0.000025, and entropy coefficient 0.001. Its independent greedy result was 0/5, so the final 1-3 model retains the behavior-cloned weights. The search, cloning, and validation implementations are in `scripts/`; their configuration and evidence inputs must match the run being performed.

### Run a new search and cloning experiment

These commands run in the foreground. Search is limited to World 1-3 and the seven-action space; it may not find a verified route within the budget. The trajectory length and result depend on the run.

```bash
.venv/bin/python scripts/explore_own_prefixes.py --portable \
  --output runs/search-seed43 --seed 43 --env-seed 0 --budget 500000 --max-hours 1.5
```

Check that the search manifest has `trajectory_verified: true` before preparing behavior cloning. Preparation creates a registration and does not train or run a game. Run the second command promptly: the deadline is established during preparation and is not refreshed when training starts.

```bash
.venv/bin/python scripts/train_own_trajectory_bc.py \
  --prepare-from-search runs/search-seed43 --registration registrations/bc-seed43.json \
  --output runs/bc-seed43 --device mps --max-hours 0.75
.venv/bin/python scripts/train_own_trajectory_bc.py --registration registrations/bc-seed43.json
```

The cloning settings use learning rate 0.001, batch size 64, gradient norm limit 0.5, up to 500 epochs, and five consecutive perfect training-set checks as the stopping condition. A fit to the recorded trajectory does not guarantee a game clear. Prepare and run a separate five-episode CPU validation:

```bash
.venv/bin/python scripts/validate_own_trajectory_policy.py \
  --prepare-from-bc runs/bc-seed43 --registration registrations/validate-seed43.json \
  --output outputs/bc-seed43-validation --max-hours 0.25
.venv/bin/python scripts/validate_own_trajectory_policy.py \
  --registration registrations/validate-seed43.json --audit-only
.venv/bin/python scripts/validate_own_trajectory_policy.py \
  --registration registrations/validate-seed43.json
```

The audit-only command checks inputs without playing; the final command runs the five games. As with cloning, the validation deadline starts at preparation, so run it promptly. `--deadline` can optionally set an explicit deadline for preparation or search.
