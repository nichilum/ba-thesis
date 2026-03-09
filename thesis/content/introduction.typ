#import "/thesis/utils/todo.typ": TODO

= Introduction

// #TODO[ // Remove this block
//   *Introduction*
//   - Introduce the reader to the general setting (No Problem description yet)
//   - What is the environment?
//   - What are the tools in use?
//   - (Not more than 1/2 a page)
// e.g. with a little historical overview.
// ]

Reverberation is apparent in most audio signals as it is an inherent characteristic of recording environments.
It is caused by late reflections ($>50-80$ ms) that overlap with the direct sound in the diffuse sound field @kuhn-rahloffSchallRaumUnd2025.
Studies have shown that reverberation is an important auditory cue which informs the listener over environmental factors @traerStatisticsNaturalReverberation2016. Depending on the application reverberation can be an attractive addition to the auditory signal, such as in music @NAYLOR2014879 or speech performances.

The inverse task aptly named dereverberation, is a process of removing reverberant parts of a recorded audio signal under reverberant conditions. Thus retaining only the direct sound of the recording. //and discarding the diffuse sound field.
This was only made feasable due to recent advancements in the field of deep learning. Historically, this task was approached using standard signal processing techniques that involved coherence estimation, suppresion rules based on thresholds and gain functions @bloomEvaluationTwoinputSpeech1982 @allenMultimicrophoneSignalprocessingTechnique1977.


// - short historical overview
//   - from filter based models to learning filter parameters to fully NN/generative based dereverberation techniques
// - dereverberation definition


In many modern applications, dereverberation is highly desirable. We divide use cases into two main categories: _offline_ and _live_ processing. Offline applications do not strictly require real-time operation, although real-time capability may still be beneficial.

== Motivation

Studies have shown that the adverse effects of reverberation mainly the temporal smearing of target speech and background noise, masking, coloration and signal-to-noise degradation @figarolaReverberationExacerbatesEffects2025 @kuhn-rahloffSchallRaumUnd2025 @cueilleEffectsReverberationSpeech2022 significantly degrade human as well as automatic (ASR or STT) speech recognition @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021.
These effects also negatively affect the overall quality of diverse audio signals such as in music remixes and film post-production, where excessive room reverberation can reduce audio clarity and limit creative flexibility.
While the above named offline applications are not in need of real-time processing, live applications, as they are used in interactive scenarios such as video conferencing, speech recognition systems, and live music performance impose strict constraints on processing latency and computational efficiency.

== Problem

Artificial reverberation can be added to audio signals with comparatively simple signal processing techniques, the inverse task of removing or reducing existing reverberation is significantly more complex @attiasSpeechDenoisingDereverberation2000. Reverberation is a time-dispersive and highly non-linear process, where direct sound and multiple delayed reflections overlap in both time and frequency @dattorroEffectDesignPart1997. This overlap makes a clear separation between the original (dry) signal and the reverberant components (wet) difficult and, for a long time, was considered practically unsolvable using classical digital signal processing methods @brandsteinUseExplicitSpeech1998.

We adress several open challenges in this thesis. First, it is unclear how well established dereverberation architectures generalize to diverse audio signals such as music and mixed content @luoTasNetTimedomainAudio2018. Current dereverberation models utilize different loss functions such as MSE, SI-SNR or PESQ @radkoffLossFunctionsAudio2021 @luoConvTasNetSurpassingIdeal2019 @lemercierStoRMDiffusionbasedStochastic2023 whose indication of dereverberation performance in diverse audio signals is not well documented. It is therefore unclear wether these loss functions are applicable for our use case or if other qualitative metrics would improve results. Second, there are time-domain and frequency-domain approaches, which can be investigated in terms of audio quality, computational complexity, and latency. Third, real-time applicability imposes strict latency limits (e.g. below 50 ms) @schmidMeasuringJustNoticeable2024 that strongly influence network architecture, window size, and sampling rate.


// Reverberation is a fundamental property of sound, but excessive or uncontrolled reverberation can significantly degrade the quality and intelligibility of speech, music, and environmental recordings. Many existing datasets lack the acoustic diversity or realism needed to evaluate modern dereverberation methods, making data collection—either through curated datasets, custom recordings, or room simulations—a crucial foundation for developing reliable algorithms. As machine-learning-based dereverberation has advanced rapidly in recent years, there is an opportunity to investigate how different model architectures perform under controlled but realistic acoustic conditions, and how data choice, simulation fidelity, and sampling rate influence model behavior.

// As machine-learning based dereverberation has advanced in recent years, works such as Conv-TasNet or StoRM have shown remarkable effectiveness in speech separation and dereverberation, yet their suitability for complex signals such as music remains unclear @luoConvTasNetSurpassingIdeal2019 @lemercierStoRMDiffusionbasedStochastic2023. At the same time, alternative architectures—both in the time domain and frequency domain—offer theoretical advantages but lack direct, systematic comparison @luoTasNetTimedomainAudio2018 @ernstSpeechDereverberationUsing2018 @luoRealtimeSinglechannelDereverberation2018.

// This thesis is motivated by the need to understand which approaches yield the highest perceptual and quantitative quality when real-time constraints are taken into account.


// Exploring the impact of sampling rate, spectral resolution, and model design we aim to provide valuable insights into how dereverberation systems can be optimized for general purpose use.

// #TODO[ // Remove this block
//   *Proposal Motivation*
//   - Outline why it is (scientifically) important to solve the problem
//   - Again use the actors to present your solution, but don't be to specific
//   - Do not repeat the problem, instead focus on the positive aspects when the solution to the problem is available
//   - Be visionary!
//   - Optional: motivate with existing research, previous work
// ]

== Objectives
// Describe the research goals and/or research questions and how you address them by summarizing what you want to achieve in your thesis, e.g. developing a system and then evaluating it.

The central objective of this thesis is to explore deep-learning-based dereverberation methods. As our primary question we ask whether a model can be designed to operate in real-time while maintaining perceptually convincing audio quality for a wide range of audio signals including music, speech and other noises (e.g. vehicle or animal sounds). Solutions for multiple performance constraints are surveyed mainly the dataset, domain and loss function.

The dataset processing and curation process is discussed regarding bias, variance and sample quality.

Time-domain and frequency-domain neural network approaches are investigated by evaluating their qualitative performance and analyzing their suitability for low-latency, real-time applications through comparison of domain specific literature and our own neural network implementations.

The performance of different quality metrics is assessed and their qualitative performance for use as loss functions examined leading to our own implementation of a loss neural network.


// == Outline
//   Describe the outline of your thesis
