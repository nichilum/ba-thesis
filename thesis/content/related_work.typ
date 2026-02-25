#import "/thesis/utils/todo.typ": TODO

= Related Work
#TODO[
  Describe related work regarding your topic and emphasize your (scientific) contribution in contrast to existing approaches / concepts / workflows. Related work is usually current research by others and you defend yourself against the statement: “Why is your thesis relevant? The problem was al- ready solved by XYZ.” If you have multiple related works, use subsections to separate them.
]

== Conv-TasNet

- builds upon tasnet (roughly describe additions)
- seperation using masking, primarly for speech signals
- encoder decoder - tcn bottleneck
- works in time-domain (not stft domain)
- realtime capable
- si-snr as loss function (we do differently)1
- noticible/audible artifacts in our tests

== StoRM

- diffusion (fully generative)

== DeepFilter

// - StoRM
// - (Conv)-TasNet
// - Wavesplit
// - DeepFilter
// - All other Models in Zotero
