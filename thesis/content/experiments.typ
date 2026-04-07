#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ
#import "/thesis/utils/diagram.typ": diagram
#import "/thesis/utils/author.typ": *

= Implementation & Experimental Setup

This section details the implementation of our three proposed networks: the perceptual quality network, the objective quality network, and the dereverberation network. The state-of-the-art model Conv-TasNet required retraining and is therefore also covered in this context.

== Conv-TasNet for diverse audio dereverberation<impl_conv_tasnet>
#leo
Conv-TasNet @luoConvTasNetSurpassingIdeal2019 operates in the time domain using a learned encoder--TCN--decoder architecture, where a temporal convolutional network estimates a multiplicative mask over the encoded signal to isolate a target source (cf. @related_work_conv_tasnet). Although originally designed for speech source separation, the masking paradigm is conceptually compatible with dereverberation. Late reflections overlap with the direct sound in the encoder representation, and a mask can in principle suppress this reverberant energy while retaining the direct component.

No pre-trained dereverberation weights were publicly available. Weights linked from the original repository were trained for speaker separation only and are thus not applicable to this task. Attempts to obtain suitable weights from the original authors received no reply. We therefore trained the model from scratch using the implementation linked in the paper #footnote[https://github.com/naplab/Conv-TasNet].

The encoder is a 1-D convolution (512 channels, window 2 ms at 8 kHz). The TCN separator consists of 8 layers across 3 stacks with feature dimension 128 and depthwise-separable convolutions of kernel size 3. The decoder mirrors the encoder via a transposed convolution. The model is configured as a single-source system. The 8 kHz sample rate imposes a hard frequency ceiling of 4 kHz, which excludes upper harmonics and air that are perceptually important for music and broadband diverse audio content @shannonCommunicationPresenceNoise1949 @isoAcousticsReferenceZero2019a @pumphreyUpperLimitFrequency1950.

The original training sets WSJ0-2mix and WSJ0-3mix @garofolojohns.CSRIWSJ0Complete2007 used in the Conv-TasNet paper are not publicly available, requiring an alternative training dataset. We used LibriSpeech @panayotovLibrispeechASRCorpus2015, resampled to 8 kHz and segmented into random 4-second crops per iteration. Reverberation is applied on-the-fly by convolving the dry signal with one of five impulse responses from the preprocessing pipeline (see @preprocessing_reverberation), yielding time-aligned wet/dry pairs with varied room conditions across epochs. The training dataset is therefore speech-only, which biases the mask priors toward speech characteristics and is expected to reduce generalization to music and other diverse audio content.

Training used the Adam optimizer @kingmaAdamMethodStochastic2017 with a learning rate of $10^(-3)$, gradient clipping with a maximum $L_2$-norm of 5.0 @luoConvTasNetSurpassingIdeal2019, and a batch size of 32 over 100 epochs via PyTorch Lightning.

#import "@preview/neural-netz:0.3.0": draw-network

== Perceptual Quality Network<impl_percep_quality_network>
#jojo

The perceptual quality network was implemented twice. @impl_percep_qual_net_init shows the initial implementation of the perceptual quality network. It features a simple encoder network and prediction heads for each scoring metric.

A second implementation based on CNN14 as introduced by #cite(<kongPANNsLargeScalePretrained2020>, form: "prose", style: "chicago-author-date") was written to address the shortcomings of the first implementation as mentioned in @eval_percep_qual_net_init.

=== Simple @CNN:short<impl_percep_qual_net_init>

The initial implementation of the perceptual quality network is based on a simple two-dimensional @CNN. This architecture was chosen because
@CNN:pl have been widely adopted in audio machine learning @grau-haroComprehensiveEvaluationCNNBased2025 and have shown great performance. The small computational cost increase as compared to a waveform-domain approach was a nonissue, as this network was not to be used during inference but only during training of the dereverberation model (see @impl_derev_net).

The forward pass includes conversion into a log-magnitude spectrogram using the @STFT, logarithmic compression is discussed in @impl_percep_qual_net_cnn14, a shared encoder counting three two-dimensional convolutional layers all featuring batch normalization, @ReLU as the activation function, and Max Pooling.

The output of this shared encoder is then fed into three prediction heads, each corresponding to one of the three initial labels (wetness, size, @ODG). The output of each prediction head is concatenated with the shared encoder output and fed into the quality prediction head, giving this model the ability to predict all parameters at once.

This gives us a better chance at debugging (cf. @eval_percep_qual_net_cnn14) and the opportunity to use just one of the four predictions as a loss function. In the end, only the quality prediction was utilized.

#diagram(caption: [Architecture of the initial implementation of the perceptual quality network], table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  stroke: 0.5pt,

  table.cell(colspan: 3)[*Perceptual Quality Network (Initial)*],

  table.cell(colspan: 3)[
    Log-magnitude spectrogram \
    STFT: n\_fft=2048, hop\_length=512
  ],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 7 times 7 @ 32; "BN, ReLU")$
    ]
  ],
  table.cell(colspan: 3)[MaxPool $2 times 2$],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 5 times 5 @ 64; "BN, ReLU")$
    ]
  ],
  table.cell(colspan: 3)[MaxPool $2 times 2$],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 128; "BN, ReLU")$
    ]
  ],
  table.cell(colspan: 3)[AdaptiveAvgPool $(4 times 4)$],

  table.cell(colspan: 3)[
    Flatten \ FC $2048 arrow.r 256$, ReLU, Dropout(0.3)
  ],

  [*ODG Head* \ FC 256 #sym.arrow 64 \ ReLU \ FC 64 #sym.arrow 1 \ Sigmoid],
  [*Size Head* \ FC 256 #sym.arrow 32 \ ReLU \ FC 32 #sym.arrow 1 \ Sigmoid],
  [*Wetness Head* \ FC 256 #sym.arrow 32 \ ReLU \ FC 32 #sym.arrow 1 \ Sigmoid],

  table.cell(colspan: 3)[
    Concat [features(256) ∥ odg(1) ∥ size(1) ∥ wetness(1)] → 259
  ],
  table.cell(colspan: 3)[
    *Quality Head* \ FC 259 #sym.arrow 64, ReLU \ FC 64 #sym.arrow 1, Sigmoid
  ],
))<arch_impl_qual_net_init>

@arch_impl_qual_net_init shows the architecture of the initial implementation. The number after the “@” symbol indicates the number of feature maps. AdamW was used as an optimizer @loshchilovDecoupledWeightDecay2019 with a learning rate of $10^(-3)$ and a weight decay value of $10^(-2)$. A per-prediction-head loss was calculated using the @MSE. The total loss was defined as:

$
  "loss" = 2 dot "loss"_"quality" + "loss"_"odg" + 0.75 dot "loss"_"size" + 0.75 dot "loss"_"wetness"
$<percep_qual_loss_init>
.
=== CNN14<impl_percep_qual_net_cnn14>

Compared to the initial implementation, this version offers a number of improvements. Mainly a new shared encoder architecture based on the CNN14 network (cf. @arch_impl_qual_net_cnn14), which was introduced as a real-time audio pattern recognition model by #cite(<kongPANNsLargeScalePretrained2020>, form: "prose", style: "chicago-author-date"). We thought it fitting as we were trying to solve an adjacent problem (audio characteristic recognition) with good performance, as we did not want to needlessly slow down the training process of the dereverberation network.

A second improvement has been made in the spectrogram conversion. The log-magnitude spectrogram was replaced with a log-mel spectrogram, offering a perceptually oriented base for the encoder. The mel scale compresses higher frequencies more than lower ones. Therefore mimicking human perception of audio. The conversion from hertz into mels is defined as @oshaughnessySpeechCommunicationHuman1987
$ m=2595 dot log_10 (1+f/700) $
. The mel scale was chosen in favor of the bark scale as it is the most used and best-performing scale in computational acoustics (e.g., in @ASR @simonkingBarkScaleVd @dhondePerformanceEvaluationMel2019).
Prediction heads as well as loss calculations were not subject to change.

The logarithmic compression of the mel scale (similar to the compression in @impl_percep_qual_net_init) is motivated by the human perception of loudness. Logarithmic compression also helps in the representation of the mel value range for use in neural network training. Without any compression, the mel scale is condensed in a small range with extreme outliers. This means the network must be trained with higher numerical precision, hence making it more susceptible to noise. It was shown that decibel-scaled melspectrograms always outperform the linear versions @choiComparisonAudioSignal2017.

Weights of the shared encoder are initialized using the Xavier uniform distribution as described in @glorotUnderstandingDifficultyTraining.

#diagram(caption: [Architecture of the CNN14 based implementation of the perceptual quality network], table(
  columns: (1fr, 1fr, 1fr),
  align: center,
  stroke: 0.5pt,

  table.cell(colspan: 3)[*Perceptual Quality Network (CNN14)*],

  table.cell(colspan: 3)[
    Log-mel spectrogram \
    sr=44100, n\_fft=2048, hop=512, n\_mels=128
  ],

  table.cell(colspan: 3)[BN (128 mel bins)],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 64; "BN, ReLU") times 2$
    ]
  ],
  table.cell(colspan: 3)[AvgPool $2 times 2$, Dropout(0.2)],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 128; "BN, ReLU") times 2$
    ]
  ],
  table.cell(colspan: 3)[AvgPool $2 times 2$, Dropout(0.2)],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 256; "BN, ReLU") times 2$
    ]
  ],
  table.cell(colspan: 3)[AvgPool $2 times 2$, Dropout(0.2)],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 512; "BN, ReLU") times 2$
    ]
  ],
  table.cell(colspan: 3)[AvgPool $2 times 2$, Dropout(0.2)],

  table.cell(colspan: 3)[
    #math.equation(block: true, numbering: none)[
      $mat(delim: "(", 3 times 3 @ 1024; "BN, ReLU") times 2$
    ]
  ],
  table.cell(colspan: 3)[AvgPool $2 times 2$, Dropout(0.2)],

  table.cell(colspan: 3)[
    Global pooling (mean over freq.) \
    Max + Mean pool over time #sym.arrow 1024-dim \
    Dropout(0.5)
  ],

  table.cell(colspan: 3)[
    FC $1024 #sym.arrow 512$, ReLU, Dropout(0.3)
  ],

  [*ODG Head* \ FC 512 #sym.arrow 128 \ ReLU, Dropout(0.3) \ FC 128 #sym.arrow 1 \ Sigmoid],
  [*Size Head* \ FC 512 #sym.arrow 64 \ ReLU, Dropout(0.3) \ FC 64 #sym.arrow 1 \ Sigmoid],
  [*Wetness Head* \ FC 512 #sym.arrow 64 \ ReLU, Dropout(0.3) \ FC 64 #sym.arrow 1 \ Sigmoid],

  table.cell(colspan: 3)[
    Concat [features(512) ∥ odg(1) ∥ size(1) ∥ wetness(1)] #sym.arrow 515
  ],

  table.cell(colspan: 3)[
    *Quality Head* \ FC 515 #sym.arrow 128, ReLU, Dropout(0.3) \ FC 128 #sym.arrow 1, Sigmoid
  ],
))<arch_impl_qual_net_cnn14>

== Objective Quality Network<impl_objective_quality_network>

@ODG was shown to worsen the performance of the perceptual quality network (see @eval_percep_qual_net_cnn14). Consequently, the perceptual quality network, as seen in @impl_percep_qual_net_cnn14, was adjusted to base the prediction only on the wetness and size parameters.

This change was done in two stages. Firstly, only the quality score was modified by removing the @ODG score (cf. @meth_obj_quality_net). Loss calculation was left unchanged. The second stage also changed the loss calculation from @percep_qual_loss_init to just

$ "loss" = "loss"_"quality" $

Both stages are evaluated in @eval_objective_quality_net.

== Dereverberation Network<impl_derev_net>
#leo

A first implementation of the dereverberation network was based on a fully generative approach, consisting of an encoder, a @TCN, and a decoder as in Conv-TasNet but without the masking operation. The decoder was expected to generate the dereverberated signal directly from the @TCN representation. However, this approach proved to be unfeasible with low computational cost, eventually requiring higher parameter counts. The model was able to overfit on the training data but did not generalize well to the validation set, as shown in @eval_derev_net. We therefore switched to a masking approach, more in line with the original Conv-TasNet architecture.

This model operates directly on waveforms. A reverberant input signal $bold(x) in RR^L$ (where $L$ is the signal length) is first projected into a learned latent space by a strided 1-D convolution $bold(W)_"enc"$, processed by a @TCN:short that predicts a real-valued mask, and finally reconstructed by a transposed convolution $bold(W)_"dec"$ (cf. @fig_forward_pass):

$
  tilde(bold(x)) = bold(W)_"dec"^top * (sigma("TCN"(bold(W)_"enc" * bold(x))) dot.o (bold(W)_"enc" * bold(x)))
$

The sigmoid-gated mask $sigma(dots) in (0,1)^(C times T)$ suppresses reverberation in the encoder feature space before the decoder maps the filtered representation back to a waveform estimate $tilde(bold(x))$.

#import "@preview/fletcher:0.5.7" as fletcher: diagram as fletcher-diagram, edge, node

#let block(pos, label, color) = node(
  pos,
  align(center, text(size: 7pt, label)),
  width: 25mm,
  height: 7mm,
  fill: color.lighten(60%),
  stroke: color.darken(10%) + 0.5pt,
  corner-radius: 3pt,
)

#let circ(pos, text) = node(
  outset: 0pt,
  inset: 0pt,
  pos,
  circle(text, radius: 10pt, stroke: 1pt + black),
)

#diagram(
  fletcher-diagram(
    spacing: (15pt, 6pt),
    cell-size: (10mm, 6mm),
    edge-stroke: 0.7pt,
    edge-corner-radius: 4pt,

    circ((0, 0), [$bold(x)$]),
    edge("-|>"),
    block((1, 0), [Encoder $bold(W)_"enc" * x$], rgb("#7B61FF")),
    edge("-|>"),
    // node((2, 0), circle(radius: 5pt, fill: rgb("#FEF3C7"), stroke: rgb("#D97706") + 0.7pt)[+]),
    node((2, 0), circle(radius: 2pt, fill: rgb("#D97706"), stroke: rgb("#D97706") + 0.7pt)),
    edge("-|>"),
    edge((2, 0), (3, 0), "-|>"),
    block((2, -2), [TCN], rgb("#0D9488")),
    edge("-|>"),
    block((3, -2), [$M = sigma(dot)$], rgb("#0D9488")),
    edge("-|>"),
    block((3, 0), [$E dot.o M$], rgb("#D97706")),
    edge("-|>"),
    block((4, 0), [Decoder $bold(W)_"dec"^top * dot$], rgb("#7B61FF")),
    edge("-|>"),
    circ((5, 0), [$tilde(bold(x))$]),
  ),
  caption: [Forward pass of the dereverberation model.],
)<fig_forward_pass>

As a loss function, the dereverberation network can either use the perceptual quality network or the @SI-SNR. As shown in @meth_silent_parts, a per-sample non-silent mask was generated. For both the @SI-SNR and the quality network loss function, it is multiplied element-wise with both the ground truth and predicted signal.

Using the perceptual (see @meth_percep_quality_net, @impl_percep_qual_net_cnn14) or objective (see @meth_obj_quality_net, @impl_objective_quality_network) quality network as loss functions requires little architectural change from the other metrics described in @analyze_loss_functions.

As the forward pass of a neural network is only a series of computations, all data requiring gradient calculation that is fed through the model will have the associated autograd graph built. To disable gradient calculation for the weights of the loss model, the gradient requirement for each parameter of the loss model is disabled. It is important to note that this should not be done using a global environment, as then all gradient calculations, including those of the dereverberation network's weights, will cease.

As explained in @impl_percep_qual_net_init, only the quality prediction was used. The final loss calculation for the dereverberation network is defined as:

$ "loss" = 1.0 - "quality" + alpha dot "MSE"_"loss" $

where quality is the predicted quality score described in @meth_percep_quality_net, $alpha$ is some factor between $0$ and $1$, and $"MSE"_"loss"$ is the @MSE between $s$ and $hat(s)$ (cf. @fun_mae_mse).

//@tab_derev_hparams summarizes all dereverberation hyperparameter configurations. //Batch-related values are reported as configuration values.

#diagram(
  caption: [Architecture of the dereverberation network (N=encoder channels, X=num blocks per repeat, R=num repeats)],
  short-caption: [Architecture of the dereverberation network],
  table(
    columns: (1fr, 1fr, 1fr),
    align: center,
    stroke: 0.5pt,

    table.cell(colspan: 3)[*Dereverberation Network (TCN v4)*],

    table.cell(colspan: 3)[
      Waveform input \
      sr=44100, segment length = 4 s
    ],

    [*Encoder*], [*Separator (TCN)*], [*Decoder*],

    [1-D Conv encoder \ $N=512$, win=$2$ ms],
    [24 residual TCN blocks \ channels=$179$, kernel size=$3$],
    [1-D transposed Conv decoder \ waveform reconstruction],

    [$E = bold(W)_"enc" * bold(x)$],
    [Dilations: $[1,2,4,8,16,32,64,128] times 3$ \ $X=8$, $R=3$],
    [$tilde(bold(x))$],

    [Masking path],
    [$M = sigma("TCN"(E))$],
    [$hat(E) = E dot.o M$],

    table.cell(colspan: 3)[
      Non-causal, layer_norm, ReLU, dropout=$0.0$, lookahead=$0$, skip connections enabled
    ],
  ),
)<arch_impl_derev_tcn_v4>

//@arch_impl_derev_tcn_v4 shows the derev_tcn_v4 architecture used for the final masking-based dereverberation experiments.



As shown in @arch_impl_derev_tcn_v4 our dereverberation network follows the core Conv-TasNet principle of encoder--masking--decoder processing but is adapted to a different task and operating regime than the original model by #cite(<luoConvTasNetSurpassingIdeal2019>, form: "prose", style: "chicago-author-date"). In contrast to the speech-separation setting of Conv-TasNet (multi-speaker mixtures at 8 kHz with permutation-invariant @SI-SNR optimization), our configuration targets single-source dereverberation on paired wet/dry signals at 44.1 kHz. The structural backbone remains comparable, including the dilation pattern with $X=8$, $R=3$, and kernel size $3$, while key capacity choices are shifted for broadband dereverberation. The encoder width is kept at $N=512$ with a 2 ms analysis window, but the TCN channel width is reduced to $179$, as shown in @tab_derev_hparams (derev_tcn_v4), to align with Conv-TasNet's 5.1 million parameter count. In addition, optimization is not restricted to @SI-SNR but is evaluated with both @SI-SNR and perceptual/objective quality-based losses, reflecting the broader quality criteria required for diverse audio content.


