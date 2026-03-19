#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/author.typ": *

= Related Work<related_work>
// #TODO[
//   Describe related work regarding your topic and emphasize your (scientific) contribution in contrast to existing approaches / concepts / workflows. Related work is usually current research by others and you defend yourself against the statement: “Why is your thesis relevant? The problem was al- ready solved by XYZ.” If you have multiple related works, use subsections to separate them.
// ]
//


== Conv-TasNet<related_work_conv_tasnet>
#leo
Conv-TasNet extends the time-domain audio separation paradigm introduced by TasNet @luoTasNetTimedomainAudio2018. TasNet was proposed as an alternative to short-time Fourier transform based source separation, motivated by the phase--magnitude decoupling of spectral methods, the fixed analysis representation, and the latency introduced by time-frequency decomposition. Instead of predicting masks on a spectrogram, TasNet learns an encoder--decoder representation directly on the waveform and performs separation in this latent space. Conv-TasNet retains this general idea, but replaces the original recurrent separator with a fully convolutional architecture designed to model long temporal context more efficiently @luoConvTasNetSurpassingIdeal2019.

Architecturally, Conv-TasNet consists of a learned linear encoder, a masking network, and a linear decoder. The encoder transforms the input waveform into a latent representation, the separator estimates multiplicative masks for this representation, and the decoder reconstructs the target waveform from the masked features. The key architectural contribution of Conv-TasNet is the use of a @TCN:both with stacked dilated one-dimensional convolution blocks as the separation module (see @fun_tcn). This allows the model to capture long-range temporal dependencies while remaining compact and fully convolutional. In the original paper, the model is trained with @SI-SNR and evaluated on single-channel, speaker-independent speech separation, where it outperforms preceding time-frequency masking approaches and even surpasses ideal time-frequency magnitude masking on the WSJ0 benchmark @luoConvTasNetSurpassingIdeal2019.

For this thesis, Conv-TasNet is relevant because it combines strong reported speech-domain performance with low model complexity and a comparatively small minimum latency, making it a plausible candidate for real-time capable dereverberation systems. At the same time, the scope of the original work remains limited to speech separation. The paper does not investigate dereverberation on diverse broadband material such as music, environmental sounds, or mixed acoustic scenes. Therefore Conv-TasNet here serves  not as a solved answer to the research problem, but as a strong real-time speech-domain baseline whose transferability to diverse-signal dereverberation must be evaluated separately.

== StoRM
#leo
StoRM is a diffusion-based stochastic regeneration model for speech enhancement and dereverberation @lemercierStoRMDiffusionbasedStochastic2023. In contrast to purely predictive enhancement systems, it follows a fully generative formulation in which the target signal is refined through a reverse diffusion process. The model combines a predictive estimate with stochastic generative refinement, aiming to retain the robustness of predictive methods while benefiting from the higher sample quality often associated with diffusion-based generation.

Architecturally, StoRM relies on score-based estimation during the reverse diffusion process and uses a predictive model as a guide for the generative reconstruction. In this sense, the score estimation component is conceptually related to the scoriing used in this thesis, although it serves a different role than the perceptual loss network introduced later. The main motivation of the original paper is to reduce artifacts that may arise in purely generative diffusion systems while still producing very clean speech restoration results. At the same time, the paper explicitly notes the high computational burden of diffusion-based inference, since multiple reverse steps are required instead of a single forward pass.

For this thesis, StoRM is relevant because it demonstrates that a generative speech-domain approach can achieve strong dereverberation quality under its intended conditions. However, the scope of the original work remains limited to speech signals and a 16 kHz setting, which limits the representable spectrum to 8 kHz by the Nyquist theorem, which is substantially below the upper range of human hearing and therefore does not preserve the full audible bandwidth of music and other broadband audio material @shannonCommunicationPresenceNoise1949 @AcousticsNormalEqualloudnesslevel2023 @zwickerSubdivisionAudibleFrequency1961. In addition, the model is not designed around strict real-time constraints, making it less suitable as a direct answer to the low-latency objective of this thesis. StoRM therefore serves here not as a solved solution to real-time dereverberation of diverse audio, but as a high-quality speech-domain reference whose transferability to broader signals and latency-constrained applications must be evaluated separately.

== DeepFilterNet<related_deep_filter>
#jojo

DeepFilterNet is a two stage speech enhancement framework utilizing deep filtering @schroterDeepFilterNetLowComplexity2022. Deep filtering is based on the idea that a neural network estimates a complex mask which is applied to the @STFT representation of a signal. With an appropriate loss function this mask can be learned to perform source extractions or other tasks like dereverberation @mackDeepFilteringSignal2020.

DeepFilterNet expands on that idea: "Instead of using a complex mask that is applied per TF-bin, [...] a combination of real-valued gains and a deep filter enhancement component [is used]." @schroterDeepFilterNetLowComplexity2022. Given a noisy signal DeepFilterNet first transforms it into the frequency-domain using the @STFT. Sampling rates up to 48 kHz and window sizes between 5 ms and 30 ms are supported. Using the frequency-domain representation @ERB features mimicing human perception are computed. Both the complex features of the @STFT and the @ERB features are used as inputs for the model. A mask is trained on the complex @STFT values and an encoder--decoder network is trained on the @ERB featues which is later used to weigh the complex mask. DeepFilterNet takes advantage from the fact that noise as well as speech usually have a smooth spectral envelope. This allows for a computationally cheap encoder--decoder network.

For this thesis, DeepFilterNet is relevant because is shows how low complexity speech enhancement can be implemented in the frequency-domain. Regarding audio sample rate the scope of this project even superceeds ours but latency is still an issue as window sizes smaller than 5 ms are not supported. As the architecture of DeepFilterNet exploits the spectral properites of speech, adaptation for use with diverse audio is difficult.


== Quality Net<related_quality_net>
#jojo
- similar approach to our loss network
  - but based and validated on pesq
  - not only BAD (high mse values)
  - but also different metric

// - StoRM
// - (Conv)-TasNet
// - Wavesplit
// - DeepFilter
// - All other Models in Zotero
