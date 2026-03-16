#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ

= Implementation & Experimental Setup

== Conv-TasNet for diverse audio dereverberation<impl_conv_tasnet>

Conv-TasNet @luoConvTasNetSurpassingIdeal2019 operates in the time domain using a learned encoder--TCN--decoder architecture, where a temporal convolutional network estimates a multiplicative mask over the encoded signal to isolate a target source (cf. @related_work_conv_tasnet). Although originally designed for speech source separation, the masking paradigm is conceptually compatible with dereverberation. Late reflections overlap with the direct sound in the encoder representation, and a mask can in principle suppress this reverberant energy while retaining the direct component.

No pre-trained dereverberation weights were publicly available. Weights linked from the original repository were trained for speaker separation only and are thus not applicable to this task. Attempts to obtain suitable weights from the original authors received no reply. We therefore trained the model from scratch using the implementation linked in the paper #footnote[https://github.com/naplab/Conv-TasNet].

The encoder is a 1-D convolution (512 channels, window 2 ms at 8 kHz). The TCN separator consists of 8 layers across 3 stacks with feature dimension 128 and depthwise-separable convolutions of kernel size 3. The decoder mirrors the encoder via a transposed convolution. The model is configured as a single-source system. The 8 kHz sample rate imposes a hard frequency ceiling of 4 kHz, which excludes upper harmonics and air that are perceptually important for music and broadband diverse audio content.

The original training sets WSJ0-2mix and WSJ0-3mix @garofolojohns.CSRIWSJ0Complete2007 used in the Conv-TasNet paper are not publicly available, requiring an alternative training dataset. We used LibriSpeech @panayotovLibrispeechASRCorpus2015, resampled to 8 kHz and segmented into random 4-second crops per iteration. Reverberation is applied on-the-fly by convolving the dry signal with one of five impulse responses from the preprocessing pipeline, yielding time-aligned wet/dry pairs with varied room conditions across epochs. The training dataset is therefore speech-only, which biases the mask priors toward speech characteristics and is expected to reduce generalization to music and other diverse audio content.

Training used the Adam optimizer @kingmaAdamMethodStochastic2017 with a learning rate of $10^(-3)$, gradient clipping with maximum $L_2$-norm of 5.0 @luoConvTasNetSurpassingIdeal2019, and a batch size of 32 over 100 epochs via PyTorch Lightning.

#import "@preview/neural-netz:0.3.0": draw-network

== Perceptual Quality Network<impl_percep_quality_network>

The perceptual quality network was implemented twice. @impl_percep_qual_net_init shows the initial implementation of the perceptual quality network. It features a simple encoder network and prediction heads for each scoring metric.

A second implementation based on CNN14 as introduced by #cite(<kongPANNsLargeScalePretrained2020>, form: "prose", style: "chicago-author-date") was written to adress the shortcomings of the first implementation as mentioned in @eval_percep_qual_net_init.

=== Initial Implementation<impl_percep_qual_net_init>

The initial implementation of the perceptual quality network is based on a simple two dimensional @CNN. A @CNN architecture was chosen because faster than real time performance for use as a loss function was not of importance. It was therefore possible to introduce a spectogram conversion.
@CNN:pl have also been widely adopted in audio machine learning @grau-haroComprehensiveEvaluationCNNBased2025.


#figure(caption: [Architecture of the initial implementation of the perceptual quality network], table(
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

@arch_impl_qual_net_init shows the architecture of the inital implementation. The number after the “@” symbol indicates the number of feature maps. Separate prediction heads for each quality metric (size, wetness, @ODG and the quality score) are suggested.
AdamW was used as an optimizer with a learning rate of $10^(-3)$.

A per-prediction-head loss was calculated head using @MSE. The total loss was defined as:

$
  "loss" = 2 dot "loss"_"quality" + "loss"_"odg" + 0.75 dot "loss"_"size" + 0.75 dot "loss"_"wetness"
$<percep_qual_loss_init>
.
=== CNN14<impl_percep_qual_net_cnn14>

A number of improvements have been made:
-

- why nn as loss (better score for perceptual, combines perceptual and "real world" attribs)
- why mel scale not bark etc.
go through loss network and explain weights (quality, size, wetness, odg) etc. make links to how data was processed for this task

- cite similar papers in zotero loss subcollection (like LEAN, etc.) for fast audio classification
  - why our loss model was based on CNN14
  - runtime (inference) evaluation


LOSS Net is based on CNN14 as shown in PANNs paper. Originally for near real time audio tagging => made sense to use here.

- same total loss as @percep_qual_loss_init
#figure(caption: [Architecture of the CNN14 based implementation of the perceptual quality network], table(
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
))


== Objective Quality Network<impl_objective_quality_network>

- two stages
  - only change quality score
  - change quality score and loss

== Dereverberation Network<impl_derev_net>
- it was shown that modifying the Conv TasNet TCN based architecture for a fully generative approach (no mask, but generate the final audio from the TCN representation) is not feasable with low computational cost (overfittable but doesn't generalize well)
  - show plots

#TODO[
  - which versions do the want to show the implementation of? only the best one (then compare si-snr with perceptual?)
  - architecture is mostly the same for all
  - show table with all hyperparameters (learning rate, batch size, etc.) for all versions
]

*inverse estimation in encoder space*
- frequency
- time (ConvTasNet)
  - use conv tasnet mask
  - test for diverse audio signals
  - compare mse to perceptual loss


#figure(
  caption: [Prediction quality of perceptual net from signal with increasing zero percentage],
  image("/experiments/perceptual-quality/plots/perceptual_net_zeros_preds.svg"),
)

#TODO[https://typst.app/universe/package/neural-netz]


== Placeholders
=== SOMETHING WITH SEGMENT LENGTH<segment_length>
=== LOSS FUNCTION SILENT MASK<loss_function_silent_mask>
