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
== Quality Metrics
- @SI-SNR
- @PEAQ and @PESQ
- @ViSQOL
- pemoq
- correlation (is this a good metric for audio signals?)
  - we used it for loss

=== PEAQ<fun_peaq>
