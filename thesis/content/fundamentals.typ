#import "/thesis/utils/author.typ": *
= Theoretical Background
== Reverberation in Active Acoustics Systems
- explain the theoretical background of coloration artifacts in live systems utilizing active acoustics when reverberation is present in a feedback loop

In these cases, reverberation can significantly degrade speech intelligibility, introduce unwanted coloration, and negatively affect the overall user experience @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021.
== TCN RNN CNN etc
#leo

The idea of autoencoders has been part of the historical landscape of neuralnetworks for decades (LeCun, 1987; Bourlard and Kamp, 1988; Hinton and Zemel,1994). @goodfellowDeepLearning2016

- short historical overview and comparison over networks, deep learning and CNNs

TCN: 
- Describe Architecture
- Usecases
- Advantages and Disadvantages over CNNs and RNNs

== LOSS
- in general training of neural net with loss function:
  - partial derivatives, gradient, jacobi matrix (analytical)
  - gradient descent explaination
- how does autograd (backward propagation work)
  - how to use this with nn as loss
- what does loss even do
== Quality Metrics
- sisnr
- peaq & pesq
- visqol
- pemoq
- correlation (is this a good metric for audio signals?)
  - we used it for loss
