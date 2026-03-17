#import "/thesis/utils/todo.typ": TODO

= Related Work<related_work>
// #TODO[
//   Describe related work regarding your topic and emphasize your (scientific) contribution in contrast to existing approaches / concepts / workflows. Related work is usually current research by others and you defend yourself against the statement: “Why is your thesis relevant? The problem was al- ready solved by XYZ.” If you have multiple related works, use subsections to separate them.
// ]
//


== Conv-TasNet<related_work_conv_tasnet>

Conv-TasNet extends the time-domain audio separation paradigm introduced by TasNet @luoTasNetTimedomainAudio2018. TasNet was proposed as an alternative to short-time Fourier transform based source separation, motivated by the phase--magnitude decoupling of spectral methods, the fixed analysis representation, and the latency introduced by time-frequency decomposition. Instead of predicting masks on a spectrogram, TasNet learns an encoder--decoder representation directly on the waveform and performs separation in this latent space. Conv-TasNet retains this general idea, but replaces the original recurrent separator with a fully convolutional architecture designed to model long temporal context more efficiently @luoConvTasNetSurpassingIdeal2019.

Architecturally, Conv-TasNet consists of a learned linear encoder, a masking network, and a linear decoder. The encoder transforms the input waveform into a latent representation, the separator estimates multiplicative masks for this representation, and the decoder reconstructs the target waveform from the masked features. The key architectural contribution of Conv-TasNet is the use of a @TCN:both with stacked dilated one-dimensional convolution blocks as the separation module (see @fun_tcn). This allows the model to capture long-range temporal dependencies while remaining compact and fully convolutional. In the original paper, the model is trained with @SI-SNR and evaluated on single-channel, speaker-independent speech separation, where it outperforms preceding time-frequency masking approaches and even surpasses ideal time-frequency magnitude masking on the WSJ0 benchmark @luoConvTasNetSurpassingIdeal2019.

For this thesis, Conv-TasNet is relevant because it combines strong reported speech-domain performance with low model complexity and a comparatively small minimum latency, making it a plausible candidate for real-time capable dereverberation systems. At the same time, the scope of the original work remains limited to speech separation. The paper does not investigate dereverberation on diverse broadband material such as music, environmental sounds, or mixed acoustic scenes. Therefore Conv-TasNet here serves  not as a solved answer to the research problem, but as a strong real-time speech-domain baseline whose transferability to diverse-signal dereverberation must be evaluated separately.

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

== Quality Net<related_quality_net>

- similar approach to our loss network
  - but based and validated on pesq
  - not only BAD (high mse values)
  - but also different metric

// - StoRM
// - (Conv)-TasNet
// - Wavesplit
// - DeepFilter
// - All other Models in Zotero
