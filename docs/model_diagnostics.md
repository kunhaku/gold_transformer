# Gold Transformer Model Overview and Improvement Plan

## 1. Data Preparation Pipeline
- **Source ingestion**: `data.ingest.load_mt5_data` pulls ordered OHLCV records (time, open, high, low, close, tick volume) from the configured SQLite table and parses timestamps.【F:data/ingest.py†L15-L27】
- **Feature engineering**: `data.features.build_feature_frame` enriches the raw frame with price distance measures, RSI, short/long moving averages, and Bollinger bands before dropping rows with insufficient history.【F:data/features.py†L7-L45】
- **Window generation**: `data.windowing.create_sliding_windows` walks the feature matrix to build progressively longer input windows paired with shrinking forecast horizons, producing masks and group IDs so the autoregressive loop can resume predictions across related slices.【F:data/windowing.py†L13-L68】
- **Dataset assembly**: `data.datasets.build_sequence_dataset` chains these steps, and `split_train_test` groups contiguous windows by `group_id` to avoid leakage when splitting into train/test partitions.【F:data/datasets.py†L43-L73】

## 2. Training Objective and Loop
- **Loss function**: Training minimizes a masked MSE that ignores padded forecast positions by weighting errors with the target mask.【F:models/losses.py†L7-L15】
- **Autoregressive conditioning**: The custom loop in `models.train.train_model` iterates group-wise, feeding the previous prediction back into the model (`past_preds`) to mimic inference behaviour.【F:models/train.py†L17-L79】
- **Optimization setup**: The model uses Adam with the learning rate from `ModelConfig`, logging per-epoch MAE/RMSE/R² for diagnostics.【F:models/train.py†L57-L76】

## 3. Model Architecture
- **Encoder stack**: `models.transformer.TransformerEncoder` applies multi-head self-attention with residual connections, layer norm, and dropout.【F:models/transformer.py†L14-L37】
- **Recurrent transformer head**: `RecurrentTransformerModel` projects inputs to the embedding dimension, passes through the encoder stack, and uses the last timestep representation to predict the full forecast vector. During training/inference it optionally concatenates tiled past predictions to the inputs.【F:models/transformer.py†L40-L72】

## 4. Diagnosed Failure Modes
1. **Covariate shift between windows** – the sliding window generator gradually increases input length while shrinking the horizon, but resets by skipping forward `initial_input_length` steps. This discards many sequences and may bias the training distribution toward shorter forecasts; the autoregressive conditioning can then drift when longer sequences reappear with more data.【F:data/windowing.py†L21-L57】
2. **No feature scaling** – technical indicators and raw prices span very different magnitudes, yet the model trains on raw values. Larger datasets exacerbate numerical imbalance, harming convergence as the optimizer chases large-magnitude price levels.
3. **Limited regularization** – the model uses a shallow encoder stack with dropout but no weight decay, learning rate scheduling, or early stopping. As more windows arrive, the fixed learning rate can overshoot minima and the autoregressive feedback can accumulate error.
4. **Sequential (non-shuffled) updates** – iterating windows strictly in chronological order per group without batching prevents gradient averaging and makes the optimizer sensitive to regime shifts, especially when later data differ structurally from earlier samples.【F:models/train.py†L57-L74】

## 5. Recommended Improvements
1. **Introduce robust scaling**: Fit per-feature scalers (e.g., `StandardScaler` or `RobustScaler`) during dataset construction, persist metadata with the existing utilities, and transform inputs/targets before windowing. This normalizes magnitudes and stabilizes training.
2. **Revisit windowing strategy**: Generate windows with consistent input/forecast lengths (or bucket by length) and avoid skipping `initial_input_length` points when rolling forward. Alternatively, compute groups via chronological splits without dynamic length changes to ensure uniform training targets.
3. **Mini-batch training**: Convert the `SequenceDataset` to `tf.data.Dataset` objects with shuffling and batching. This reduces variance, enables vectorized computation, and allows adaptive optimizers with warm restarts or learning-rate schedulers.
4. **Regularization and monitoring**: Add validation splits, early stopping, and possibly label smoothing on the autoregressive feedback (e.g., teacher forcing with a blend of ground truth and previous predictions) to limit error accumulation.
5. **Model capacity and residual connections**: Experiment with deeper encoder stacks or add convolutional/contextual embeddings (e.g., learnable positional encodings) so the model can capture seasonal patterns without relying solely on the last timestep representation.

Implementing the scaling and batching changes typically yields the quickest wins for stability when increasing training data volume. The windowing simplification further ensures that the distribution seen during training matches inference, preventing the observed divergence as more samples are introduced.

## 6. Where the "revisit" windowing concept fits

The original revisit design assumed you would set a concrete target far in the future, then re-open the same trajectory multiple times as time advanced so the model could check whether its early hypothesis still held. That mental model is powerful for **decision-making** because it mirrors how a human trader commits to a thesis, tracks new evidence, and updates conviction. However, the current transformer agent is optimized for **supervised sequence-to-sequence forecasting**. Within that scope, every training sample must expose the ground-truth horizon immediately; there is no higher-level planner to remember the initial thesis or to request a "revisit" when new candles arrive. As a result, encoding the revisit logic directly into the dataset builder forces the forecaster to learn two incompatible tasks at once: (1) predicting a fresh horizon and (2) validating previous hypotheses midstream. This dual role makes the loss explode as more data are added because the supervision signal keeps shifting.

If you want to keep the revisit workflow, place it **one layer above** the forecasting agent:

1. Let the base transformer produce standard fixed-horizon forecasts from consistently shaped windows (the adjustments listed above).
2. Add a lightweight supervisory module—a rules engine or small policy network—that records each time a trade thesis is opened. This module decides when to request a new forecast from the transformer ("revisit"), compares it with the stored thesis, and takes action (hold, adjust stop, exit).
3. Feed the supervisory module summaries of the transformer's predictions plus any external context you track (risk budget, macro events). That hierarchy mirrors your original intuition: the higher level reasons about targets over time; the lower level supplies up-to-date probabilistic forecasts.

By separating concerns, the forecasting agent stays statistically stable, while the decision hierarchy implements the adaptive revisit behaviour you designed.

## 7. Evaluating the revisit windowing strategy itself

- **Strengths**: The revisit workflow preserves a rich narrative around each trading idea, encourages disciplined hypothesis tracking, and naturally supplies evaluation hooks—every revisit produces a labelled comparison between the prior thesis and the updated evidence. When deployed above a stable forecaster, those hooks make it easier to audit why trades succeed or fail and to refine the decision policy.
- **Limitations**: The strategy relies on a persistent memory of prior targets and contextual cues; without a supervisory layer, those signals collapse into noisy labels for the base model. It also assumes the information gain between revisits is monotonic (later windows are "better"), which may fail during regime shifts or when market microstructure noise dominates new candles.
- **Best use cases**: Treat the revisit step as a governance mechanism—trigger it when volatility spikes, when macro catalysts arrive, or when the original forecast confidence decays. Use the collected revisit traces to evaluate whether the decision hierarchy intervened at the right time, rather than as direct training data for the forecaster.

In short, the revisit windowing strategy is valuable for decision intelligence, provided it operates as a supervising process that selectively requests fresh forecasts instead of reshaping the forecaster's raw training data.

## 8. Implemented supervisory scaffold

- **Reusable forecast traces**: `evaluation.iterate_group_predictions` now exposes the same autoregressive rollout used during inference, while `generate_group_forecast` returns the full trace for a single trajectory so higher-level components can reuse the existing preprocessing and transformer model without recomputing every group.【F:evaluation/inference.py†L16-L111】
- **Decision-layer orchestration**: The new `decision.revisit` module introduces `RevisitSupervisor`, `TradeThesis`, and supporting dataclasses that store the initial forecast, revisit events, and per-step diagnostics while leaving the base transformer untouched.【F:decision/revisit.py†L22-L200】
- **Thesis summaries**: Supervisory clients can open a thesis, trigger revisits, and request a serialisable summary showing prediction deltas, realised targets, and metrics for auditing the strategy envisioned in Sections 6 and 7.【F:decision/revisit.py†L60-L223】
- **End-to-end orchestration**: `pipelines.run_revisit_workflow` now wraps dataset preparation, model training, inference, and the supervisory layer so you can execute the full pipeline (including revisit bookkeeping) with a single entry point.【F:pipelines/revisit_pipeline.py†L1-L128】【F:pipelines/training_pipeline.py†L1-L68】
