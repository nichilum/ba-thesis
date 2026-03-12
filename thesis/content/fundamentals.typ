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

#TODO[
  WHAT DO WE ACTUALLY NEED HERE?:
  SISDR
  Source-to-Artifact Ratio (SAR)
  Source-to-Interference Ratio (SIR)
  Source-to-Distortion Ratio (SDR)
  Signal-to-Noise Ratio (SNR)
]

The following section will present different quality metrics desgined for comparative analysis of two input vectors. Going forward the input vectors will be considered signals as we are examining these measures from a signal processing standpoint.

$ s $
is defined as the ground truth, also named reference or true, signal.

$ hat(s) $
is defined as the predicted, also named test or processed, signal.

All subsequent measures are investigated for general usability in audio adjacent machine learning tasks. Most are used in @results for comparative evaluation of different neural networks. A discussion of usability as a loss function for a dereverberation neural network is found in @loss_network.


=== MAE and MSE<fun_mae_mse>

The @MAE

$ "MAE" = 1/n sum_(i=1)^n (s_i - hat(s)_i) $

measures the average absolute error between to signals. The @MSE:long

$ "MSE" = 1/n sum_(i=1)^n (s_i - hat(s)_i)^2 $

measures the average squared difference between the predicted and the ground truth signal. Although both the @MSE and @MAE were used successfully as loss functions in e.g. music source separation approaches @defossezMusicSourceSeparation2019 @stollerWaveUNetMultiScaleNeural2018 @takahashiD3NetDenselyConnected2020 they fall short in generative and human-ear centered tasks as both unfairly penalize shifts in time and amplitude of the predicted signal and do not conform to the equal-loudness levels as perceived by the human ear @AcousticsNormalEqualloudnesslevel2023 and therefore overweight the importance of low frequencies.

=== Correlation<fun_corr>

The Pearson's product-momentum coefficient is defined as:

$
  rho_(Y, hat(Y)) = "corr"(Y, hat(Y))="cov"(Y, hat(Y))/(sigma_Y sigma_hat(Y)) = ("E"[(Y-mu_Y)(hat(Y)-mu_hat(Y))])/(sigma_Y sigma_hat(Y)), "if" sigma_Y sigma_hat(Y) > 0
$

where $sigma_Y "and" sigma_hat(Y)$ are the standard deviations, $mu_Y "and" mu_hat(Y)$ the expected values and $"E"$ the expected values operator @benestyPearsonCorrelationCoefficient2009. The result of the Pearson coefficient can be interpreted as seen in @p_coeff_interp:

#figure(caption: [Interpretation of the Pearson coefficient], table(
  columns: 3,
  [*$rho_(Y, hat(Y))$*], [*$rho_(Y, hat(Y))$*], [*Association Between Variables*],
  [$+0.8 "to" +1.0$], [$-0.8 "to" -1.0$], [Very strong association],
  [$+0.6 "to" +0.8$], [$-0.6 "to" -0.8$], [Strong association],
  [$+0.4 "to" +0.6$], [$-0.4 "to" -0.6$], [Moderate association],
  [$+0.2 "to" +0.4$], [$-0.2 "to" -0.4$], [Weak association],
  [$+0.0 "to" +0.2$], [$-0.0 "to" -0.2$], [Very weak or no association],
))<p_coeff_interp>


The problem is that both input signals are assumed to be two random variables which is technically not the case. Although correlation has been used successfully in computational audio tasks such as simultaneous sound event localization @cordourierGCCPHATCrossCorrelationAudio2019 using a statistical relationship to compare a reference to a test signal proved challenging (see @loss_network).

=== SI-SNR<fun_si-snr>

The @SI-SNR:long

$ "SI-SNR" = 10 log_10 ((||a s||^2)/(||a s - hat(s)||^2)), "where" a = (hat(s)^T s)/(||s||^2) $

measures the level of distortion or noise in the predicted signal in a way that is invariant to the scaling of the signals. It has been used successfully in dereverberation tasks @luoConvTasNetSurpassingIdeal2019 but while providing invariance to signal scaling it too does not conform to the perceived loudness of the human ear nor provide invariance to signal shifting.

=== PESQ<fun_pesq>

Answering the shortcoming of metrics like the @MSE and @SI-SNR, the @PESQ:both model (a successor to the @BSD and @PSQM models) is both invariant to signal scaling and shifting. It also maps the signal into a representation of percieved loudness in time and frequency through a psychoacoustic model based on the bark scale @rixPerceptualEvaluationSpeech2001 which is a psychoacoustical scale on which equal distances correspond with perceptually equal distances @zwickerSubdivisionAudibleFrequency1961 therefore assuring conformity with the human auditory system (cf. @speech_quality_pipeline).

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

=== PEAQ<fun_peaq>

The @PEAQ model is based on the @PAQM model and has been an ITU-R recommendation since 1999 @rixPerceptualEvaluationSpeech2001. It offers two metrics, namely the @ODG:both and @DI:both. The @ODG corresponds with the @SDG and indicates the audio quality of the tested signal on a continuous scale from -4 (very annoying impairment) to 0 (imperceptible impairment). The @DI is a quality indicator like the @ODG except for its higher sensitivity towards very low signal qualities @khalifehPerceptualEvaluationAudio2017 @thiedePEAQITUStandard2000.

#TODO[short text about ear model]

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

=== ViSQOL<fun_visqol>
=== PEMO-Q<fun_pemoq>
