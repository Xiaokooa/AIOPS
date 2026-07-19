# 24h-Lookback 1h-Ahead Deep Training Protocol

The official deep time-series classifiers now train an OFP-compatible
first-warning alarm head with a rolling 24h lookback and a 1h-ahead
first-event auxiliary target.

- Each valid timestamp before the first anomaly receives a 1-column target:
  `0 < first_anomaly_ts - timestamp <= 1h`.
- Rows at or after the first anomaly remain excluded from training,
  normalization, validation threshold selection, and evaluation triggers.
- Model output shape is `[alarm, h1]`.
- The `alarm` column is the primary score used for threshold selection and
  first-warning evaluation.
- iTransformer, PatchTST, ModernTCN, FITS, and FTEformer builders output two
  logits by default.
- PatchTST segment runners inherit the same 1h primary horizon from the shared
  protocol constants.
- FTEformer diagnostic/type losses use the alarm logit for event consistency,
  while the auxiliary BCE uses the 1h target.
- Deep runners now default to imbalance-aware sampling
  `train_sampling=module_balanced`: each module contributes a capped number of
  1h-positive, pre-window negative, and normal windows.
- Validation-based deep runners select the warning rule by final-score search
  over threshold, causal EMA smoothing windows, and consecutive-hit `K`.
  Prediction files can emit only the first alarm timestamp per module
  (`first_alarm_only=True` by default).
- The official trainers optimize a two-part objective:
  `event_loss_weight * first_warning_event_loss + aux_bce_weight * weighted h1 BCE`.
  The event loss maximizes the probability that the first alarm occurs inside
  the 1h-ahead window for faulty modules and that no alarm occurs for normal
  modules.

The separate semi-supervised MIL pipeline lives under
`D:\AIOps\OFP_DL_semisup_mil`.
