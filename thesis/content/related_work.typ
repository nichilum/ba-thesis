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

- diffusion architecture (fully generative)
- score function estimator (similar to our loss network)
- high computation requirements
- not realtime capable
- clean seperation results
- speech only

== DeepFilterNet

- seeks to have better performance than real-values or complex-masks (need high freq resolution)
- uses Deep Filters, that are filters applied to multiple time/freq bins
- based on CLC (complex linear coding)
- compared using sisnr to CRM (complex ratio mask): it's better
- viable for real-time usage
- 48kHz

For the first stage, we take advantage from the fact that noise as well as speech usually
have a smooth spectral envelope
=> prob not entirely possible for music and diverse audio signals



// - StoRM
// - (Conv)-TasNet
// - Wavesplit
// - DeepFilter
// - All other Models in Zotero
