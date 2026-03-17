#import "/thesis/utils/author.typ": *
#import "/thesis/utils/todo.typ": TODO
#import "@preview/diagraph:0.3.6": *

= Theoretical Background

== Acoustics


=== Reverberation

convolutions are a fast operation in the frequency domain and on GPU devices @siddiqOptimizationConvolutionReverberation2020 @misicAnalysisCPUGPU2016


=== Reverberation in Active Acoustics Systems
- explain the theoretical background of coloration artifacts in live systems utilizing active acoustics when reverberation is present in a feedback loop

In these cases, reverberation can significantly degrade speech intelligibility, introduce unwanted coloration, and negatively affect the overall user experience @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021.

== Neural Networks
Neural networks are parameterized function approximators composed of interconnected layers of simple computational units. During training, their weights and biases are iteratively adjusted so that the network maps an input to a desired output while minimizing a loss function @goodfellowDeepLearning2016. In practice, modern architectures differ mainly in how they organize these computations and which inductive bias they impose on the data. For this thesis, the most relevant families are @CNN:pl, @RNN:pl, and @TCN:pl.

=== @CNN:short

@CNN:pl are feed-forward neural networks that process structured inputs by applying learned convolution kernels over local neighborhoods. Instead of connecting every input element to every neuron, a convolutional layer reuses the same filter weights across the full input. This weight sharing reduces the number of parameters and makes the network particularly effective at detecting local patterns such as edges in images, harmonics in spectrograms, or short waveform structures @goodfellowDeepLearning2016. Stacking multiple convolution layers increases the receptive field, so deeper layers can combine simple local features into more abstract representations.

In audio machine learning, @CNN:pl are widely used on time-frequency representations because spectrograms preserve local correlations in time and frequency that can be captured efficiently by two-dimensional filters. One-dimensional variants are also common when operating directly on waveforms or learned feature sequences. Their main advantages are computational efficiency, parameter sharing, and good parallelizability on modern hardware. A limitation is that long-range temporal dependencies are not represented explicitly and must instead be captured through depth, pooling, or large receptive fields @grau-haroComprehensiveEvaluationCNNBased2025 @kongPANNsLargeScalePretrained2020.

=== @RNN:short

#TODO[add some visualizations?]

@RNN:pl are designed for sequential data. In addition to the current input, each recurrent step receives a hidden state that carries information from previous time steps, allowing the network to model temporal dependencies. This makes @RNN:pl conceptually well suited for signals, text, and time series, where the interpretation of one sample often depends on earlier context, @rumelhartLearningRepresentationsBackpropagating1986 @goodfellowDeepLearning2016. In audio applications, recurrent layers have been used for tasks such as speech enhancement, quality prediction, and sequence modeling because they can aggregate information over longer temporal spans than shallow feed-forward models.

Classical @RNN:pl suffer from vanishing and exploding gradients when the dependency horizon becomes long. For this reason, gated variants such as @LSTM @hochreiterLongShortTermMemory1997 and @GRU networks @choPropertiesNeuralMachine2014 are commonly used in practice. These architectures regulate which information is stored, updated, or forgotten, improving training stability and long-term memory. Their main drawback is that recurrent processing is inherently sequential, which limits parallelization during training and inference compared to purely convolutional models @fuQualityNetEndtoEndNonintrusive2018 @defossezMusicSourceSeparation2019.

=== @TCN:short<fun_tcn>

@TCN:pl adapt the convolutional idea to sequence modeling by applying one-dimensional convolutions along the temporal axis. To cover long contexts efficiently, they often use dilated convolutions, where filter taps are spaced apart by increasing dilation factors. This enlarges the receptive field without requiring very deep networks or large kernels. Depending on the application, @TCN:pl can be implemented causally, where each output depends only on the present and past, or non-causally, where future context is also available @baiEmpiricalEvaluationGeneric2018.

#TODO[figure 1 from @baiEmpiricalEvaluationGeneric2018 for tcn architecture]

Compared to @RNN:pl, @TCN:pl retain the ability to model long temporal structure while remaining fully convolutional and therefore highly parallelizable. They also provide explicit control over receptive field size through kernel width, depth, and dilation schedule. This makes them attractive for audio tasks that require a compromise between temporal context and computational efficiency. In this thesis, @TCN:pl are particularly relevant because Conv-TasNet uses a temporal convolutional network to estimate masks over an encoded waveform representation @luoConvTasNetSurpassingIdeal2019. The basic principles of @TCN:pl therefore form part of the architectural foundation for the dereverberation models discussed later.

=== Encoders and Decoders

=== Supervised Learning<supervised_learning>
=== Self-Supervised Learning<self_supervised>

=== Loss Function<fun_loss_function>

- Gradient Descent
- Backwardpropagation
- Autograd
- in general training of neural net with loss function:
  - partial derivatives, gradient, jacobi matrix (analytical)
  - gradient descent explaination
- how does autograd (backward propagation work)
  - how to use this with nn as loss
- what does loss even do
-
== Quality Metrics<fun_quality_metrics>

The following section will present different quality metrics desgined for comparative analysis of two input vectors. Going forward the input vectors will be considered signals as we are examining these measures from a signal processing standpoint.

/ $s$: is defined as the ground truth, also named reference or true, signal.
/ $hat(s)$: is defined as the predicted, also named test or processed, signal.

All subsequent measures are investigated for general usability in audio adjacent machine learning tasks. Most are used in @results for comparative evaluation of different neural networks. A discussion of usability as a loss function for a dereverberation neural network is found in @meth_percep_quality_net.

=== @MAE:short and @MSE:short<fun_mae_mse>

The @MAE

$ "MAE" = 1/n sum_(i=1)^n (s_i - hat(s)_i) $

measures the average absolute error between to signals. The @MSE:long

$ "MSE" = 1/n sum_(i=1)^n (s_i - hat(s)_i)^2 $

measures the average squared difference between the predicted and the ground truth signal. Although both the @MSE and @MAE were used successfully as loss functions in e.g. music source separation approaches @defossezMusicSourceSeparation2019 @stollerWaveUNetMultiScaleNeural2018 @takahashiD3NetDenselyConnected2020 they fall short in generative and human-ear centered tasks as both unfairly penalize shifts in time and amplitude of the predicted signal and do not conform to the equal-loudness levels as perceived by the human ear @AcousticsNormalEqualloudnesslevel2023 and therefore overweight the importance of low frequencies.

=== Correlation<fun_corr>

The Pearson's product-momentum coefficient is defined as:

$
  rho_(s, hat(s)) = "corr"(s, hat(s))="cov"(s, hat(s))/(sigma_s sigma_hat(s)) = ("E"[(s-mu_s)(hat(s)-mu_hat(s))])/(sigma_s sigma_hat(s)), "if" sigma_s sigma_hat(s) > 0
$

where $sigma_s "and" sigma_hat(s)$ are the standard deviations, $mu_s "and" mu_hat(s)$ the expected values and $"E"$ the expected values operator @benestyPearsonCorrelationCoefficient2009. The result of the Pearson coefficient can be interpreted as seen in @p_coeff_interp, where negative values mean inverse association:

#figure(caption: [Interpretation of the Pearson coefficient], table(
  columns: 3,
  [*$rho_(s, hat(s))$*], [*$rho_(s, hat(s))$*], [*Association Between Variables*],
  [$+0.8 "to" +1.0$], [$-0.8 "to" -1.0$], [Very strong association],
  [$+0.6 "to" +0.8$], [$-0.6 "to" -0.8$], [Strong association],
  [$+0.4 "to" +0.6$], [$-0.4 "to" -0.6$], [Moderate association],
  [$+0.2 "to" +0.4$], [$-0.2 "to" -0.4$], [Weak association],
  [$+0.0 "to" +0.2$], [$-0.0 "to" -0.2$], [Very weak or no association],
))<p_coeff_interp>


The problem is that both input signals are assumed to be two random variables which is technically not the case. Although correlation has been used successfully in computational audio tasks such as simultaneous sound event localization @cordourierGCCPHATCrossCorrelationAudio2019 using a statistical relationship to compare a reference to a test signal proved challenging (see @analyze_loss_functions).

=== @SI-SNR:short<fun_si-snr>

The @SI-SNR:long

$ "SI-SNR" = 10 log_10 ((||a s||^2)/(||a s - hat(s)||^2)), "where" a = (hat(s)^T s)/(||s||^2) $

measures the level of distortion or noise in the predicted signal in a way that is invariant to the scaling of the signals. It has been used successfully in dereverberation tasks @luoConvTasNetSurpassingIdeal2019 but while providing invariance to signal scaling it too does not conform to the perceived loudness of the human ear nor provide invariance to signal shifting.

It can also be mentioned that there are other variants, like Source-to-Artifact Ratio (SAR), Source-to-Interference Ratio (SIR), Source-to-Distortion Ratio (SDR) and Signal-to-Noise Ratio (SNR), each with Scale-Invariant (SI) forms, which are used in the field of source separation and speech enhancement, but are all inspired by the usual definition of the SNR @vincentPerformanceMeasurementBlind2006.

=== @PESQ:short<fun_pesq>

Answering the shortcoming of metrics like the @MSE and @SI-SNR, the @PESQ:both model (a successor to the @BSD and @PSQM models) is both invariant to signal scaling and shifting. The @PESQ score reflects speech quality on a continuous scale ranging from 1 to 5 (cf. @pesq_score_interp)

#figure(
  caption: [The Absolute Category Rating scale used by @MOS/@PESQ],
  table(
    columns: 2,
    table.header([*Rating*], [*Label*]),
    [5], [Excellent],
    [4], [Good],
    [3], [Fair],
    [2], [Poor],
    [1], [Bad],
  ),
)<pesq_score_interp>

The scale shown in @pesq_score_interp corresponds to the @MOS scale. During analysis the signal is mapped into a representation of perceived loudness in time and frequency through a psychoacoustic model based on the bark scale @rixPerceptualEvaluationSpeech2001 which is a psychoacoustical scale on which equal distances correspond with perceptually equal distances @zwickerSubdivisionAudibleFrequency1961 therefore assuring conformity with the human auditory system (cf. @speech_quality_pipeline).

#figure(caption: [Structure of @PESQ:both model taken from @rixPerceptualEvaluationSpeech2001], raw-render(
  ```dot
      digraph pesq {
        rankdir=LR
        splines=ortho
        node [fontsize=10, style=filled, shape=box, fillcolor="white"]
        edge [fontsize=8]
        ref_sig      [shape=plain, fillcolor=none]
        deg_sig      [shape=plain, fillcolor=none]
        level_align1
        level_align2
        input_filt1
        input_filt2
        time_align   [height=3]
        aud_trans1
        aud_trans2
        dist_proc
        cog_model
        bad_int
        output       [shape=plain, fillcolor=none]
        {rank=same; ref_sig; deg_sig}
        {rank=same; level_align1; level_align2}
        {rank=same; input_filt1; input_filt2}
        {rank=same; aud_trans1; dist_proc; aud_trans2}
        {rank=same; cog_model; bad_int}

        aud_trans1 -> dist_proc -> aud_trans2 [style=invis, weight=100]
        ref_sig   -> level_align1
        deg_sig   -> level_align2
        level_align1 -> input_filt1
        level_align2 -> input_filt2
        input_filt1  -> time_align
        input_filt2  -> time_align
        time_align   -> aud_trans1
        time_align   -> aud_trans2
        aud_trans1   -> dist_proc
        aud_trans2   -> dist_proc
        aud_trans1   -> time_align [constraint=true]
        aud_trans2   -> time_align [constraint=false]
        dist_proc    -> cog_model
        dist_proc    -> bad_int
        bad_int      -> time_align [label="Re-align bad intervals", constraint=true]
        cog_model    -> output
      }
  ```,
  labels: (
    ref_sig: [Reference signal],
    deg_sig: [Degraded signal],
    level_align1: [*Level\ align*],
    level_align2: [*Level\ align*],
    input_filt1: [*Input\ filter*],
    input_filt2: [*Input\ filter*],
    time_align: [*Time align\ and equalise*],
    aud_trans1: [*Auditory\ transform*],
    aud_trans2: [*Auditory\ transform*],
    dist_proc: [*Disturbance\ processing*],
    cog_model: [*Cognitive\ modelling*],
    bad_int: [*Identify bad\ intervals*],
    output: [*Prediction of\ perceived\ speech\ quality*],
  ),
  width: 15cm,
))<speech_quality_pipeline>


=== @PEAQ:short<fun_peaq>

The @PEAQ model is based on the @PAQM model and has been an ITU-R recommendation since 1999 @rixPerceptualEvaluationSpeech2001. In general it compares two time aligned signals, one processed and one original. Concurrent frames of each signal are transformed to a basilar membrane representation whose differences are further analyzed by a cognitive model @thiedePEAQITUStandard2000 (cf. @audio_quality_pipeline). The two offered metrics, namely the @ODG:both and @DI:both, are therefore not invariant to signal shifting but they conform to the human perception of sound loudness. The @ODG corresponds with the @SDG and indicates the audio quality of the tested signal on a continuous scale from -4 (very annoying impairment) to 0 (imperceptible impairment). The @DI is a quality indicator like the @ODG except for its higher sensitivity towards very low signal qualities @khalifehPerceptualEvaluationAudio2017.

#figure(caption: [High-level representation of the @PEAQ:both model taken from @thiedePEAQITUStandard2000], raw-render(
  ```dot
      digraph peaq {
        rankdir=TB
        splines=ortho
        node [fontsize=10, style=filled, shape=box, fillcolor="white"]
        edge [fontsize=8]
        proc_sig      [fillcolor=none]
        org_sig       [fillcolor=none]
        ear_model     [fillcolor=lightgray]
        feat_extraction [fillcolor=lightgray]
        movs
        quality

        {rank=same; movs; quality}

        proc_sig -> ear_model
        org_sig -> ear_model
        ear_model -> feat_extraction
        ear_model -> movs [constraint=false]
        feat_extraction -> movs
        feat_extraction -> movs
        feat_extraction -> movs
        feat_extraction -> movs
        feat_extraction -> quality
        movs -> quality
      }
  ```,
  labels: (
    proc_sig: [*Processed Signal*],
    org_sig: [*Original Signal*],
    ear_model: [*Peripheral Ear Model*],
    feat_extraction: [*Feature extraction and Combination*],
    movs: [*MOVs*],
    quality: [*Quality grade*],
  ),
  height: 5cm,
))<audio_quality_pipeline>
