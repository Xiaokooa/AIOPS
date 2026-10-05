# Third-party notices

The following upstream components retain their MIT licenses and copyright
notices. Only layers required by the baseline adapters are distributed here.

| Component | Source | Included license |
|---|---|---|
| iTransformer | https://github.com/thuml/iTransformer | `third_party/iTransformer/LICENSE` |
| PatchTST | https://github.com/yuqinie98/PatchTST | `third_party/PatchTST/PatchTST-main/LICENSE` |
| ModernTCN | https://github.com/luodhhh/ModernTCN | `third_party/ModernTCN/LICENSE` |

The iTransformer attention file is reduced to FullAttention and AttentionLayer;
their numerical implementation is unchanged. Local classification adapters
load these layers without changing global import namespaces.

The FITS adapter implements low-frequency complex linear filtering in PyTorch.
Related work: https://github.com/VEWOXIC/FITS
