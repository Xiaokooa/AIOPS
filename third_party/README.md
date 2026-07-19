# Third-party model sources

This directory contains upstream model implementations required by OFP model
wrappers. Keep project-specific training, evaluation, and feature logic under
`OFP/` rather than modifying these sources.

- `iTransformer`, `ModernTCN`, and `PatchTST` are vendored source snapshots.
- `timesfm` is managed as a Git submodule; initialize it with
  `git submodule update --init --recursive`.
