#import "/thesis/utils/todo.typ": TODO

= Related Work<related_work>
// #TODO[
//   Describe related work regarding your topic and emphasize your (scientific) contribution in contrast to existing approaches / concepts / workflows. Related work is usually current research by others and you defend yourself against the statement: “Why is your thesis relevant? The problem was al- ready solved by XYZ.” If you have multiple related works, use subsections to separate them.
// ]

== Conv-TasNet<related_work_conv_tasnet>

- builds upon tasnet @luoTasNetTimedomainAudio2018 (roughly describe additions)
- seperation using masking, primarly for speech signals
- encoder decoder - tcn bottleneck
- works in time-domain (not stft domain)
- realtime capable
- si-snr as loss function (we do differently)1
- noticible/audible artifacts in our tests
- 8 kHz
- @luoConvTasNetSurpassingIdeal2019

== StoRM

- diffusion architecture (fully generative)
- score function estimator (similar to our loss network)
- high computation requirements
- not realtime capable
- clean seperation results
- speech only
- 16 kHz
- @lemercierStoRMDiffusionbasedStochastic2023

== DeepFilterNet

- seeks to have better performance than real-values or complex-masks (need high freq resolution)
- uses Deep Filters, that are filters applied to multiple time/freq bins
- based on CLC (complex linear coding)
- compared using sisnr to CRM (complex ratio mask): it's better
- viable for real-time usage
- 48kHz
- @schroterDeepFilterNetLowComplexity2022

For the first stage, we take advantage from the fact that noise as well as speech usually
have a smooth spectral envelope
=> prob not entirely possible for music and diverse audio signals

== Quality Net

- similar approach to our loss network
  - but based and validated on pesq
  - not only BAD (high mse values)
  - but also different metric

// - StoRM
// - (Conv)-TasNet
// - Wavesplit
// - DeepFilter
// - All other Models in Zotero
