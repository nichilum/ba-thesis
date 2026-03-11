#import "/thesis/utils/author.typ": *
= Theoretical Background

== Acoustics


=== Reverberation

convolutions are a fast operation in the frequency domain and on GPU devices @siddiqOptimizationConvolutionReverberation2020 @misicAnalysisCPUGPU2016


=== Reverberation in Active Acoustics Systems
- explain the theoretical background of coloration artifacts in live systems utilizing active acoustics when reverberation is present in a feedback loop

In these cases, reverberation can significantly degrade speech intelligibility, introduce unwanted coloration, and negatively affect the overall user experience @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021.

== Neural Networks
=== TCN RNN CNN etc
#leo


The idea of autoencoders has been part of the historical landscape of neuralnetworks for decades (LeCun, 1987; Bourlard and Kamp, 1988; Hinton and Zemel,1994). @goodfellowDeepLearning2016

- short historical overview and comparison over networks, deep learning and @CNN:pl

@TCN:
- Describe Architecture
- Usecases
- Advantages and Disadvantages over @CNN:pl and @RNN:pl
- look at description in https://www.researchgate.net/publication/360936572_An_enhanced_Conv-TasNet_model_for_speech_separation_using_a_speaker_distance-based_loss_function
=== Supervised Learning<supervised_learning>
=== Self-Supervised Learning<self_supervised>

=== Loss Function<fun_loss_function>
- in general training of neural net with loss function:
  - partial derivatives, gradient, jacobi matrix (analytical)
  - gradient descent explaination
- how does autograd (backward propagation work)
  - how to use this with nn as loss
- what does loss even do
-
== Quality Metrics<fun_quality_metrics>

- all metrics are analyzed from an acoustical nn standpoint

=== MAE and MSE<fun_mae_mse>

The @MAE

$ "MAE" = ... $

measures the average ... . The @MSE:long

$ "MSE" = 1/n sum_(i=1)^n (Y_i - hat(Y)_i)^2 $

measures the average squared difference between the predicted values and the ground truth value. Although both the @MSE and @MAE were used successfully as loss functions in e.g. music source separation approaches @defossezMusicSourceSeparation2019 @stollerWaveUNetMultiScaleNeural2018 @takahashiD3NetDenselyConnected2020 they fall short in generative and human-ear centered tasks as both unfairly penalize shifts in time and amplitude of the predicted signal and do not conform to the equal-loudness levels as perceived by the human ear @AcousticsNormalEqualloudnesslevel2023 and therefore overweight the importance of low frequencies.

=== Correlation

=== SI-SNR<fun_si-snr>

The @SI-SNR:long

$ "SI-SNR" = 10 log_10 ((||a s||^2)/(||a s - hat(s)||^2)), "where" a = (hat(s)^T s)/(||s||^2) $

measures the level of distortion or noise in the predicted signal in a way that is invariant to the scaling of the signals. It has been used successfully in dereverberation tasks @luoConvTasNetSurpassingIdeal2019 but while providing invariance to signal scaling it too does not conform to the perceived loudness of the human ear.

=== PESQ<fun_pesq>

Answering the shortcoming of metrics like the @MSE and @SI-SNR, the @PESQ:both model (a successor to the @BSD and @PSQM models) is both invariant to signal scaling and shifting. It also maps the signal into a represantation of percieved loudness in time and frequency through a psychoacoustic model based on the bark scale @rixPerceptualEvaluationSpeech2001 which is a psychoacoustical scale on which equal distances correspond with perceptually equal distances @zwickerSubdivisionAudibleFrequency1961 therefore assuring conformity with the huamn auditory system.

=== PEAQ<fun_peaq>

The @PEAQ model is based on the @PAQM model and has been an ITU-R recommendation since 1999 @rixPerceptualEvaluationSpeech2001. It offers two metrics, namely the @ODG:both and @DI:both. The @ODG corresponds with the @SDG and indicates the audio quality of the tested signal on a continuous scale from -4 (very annoying impairment) to 0 (imperceptible impairment). The @DI is a quality indicator like the @ODG except for its higher sensitivity towards very low signal qualities @khalifehPerceptualEvaluationAudio2017 @thiedePEAQITUStandard2000.

=== ViSQOL
=== PEMO-Q
