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


- why nn as loss (better score for perceptual, combines perceptual and "real world" attribs)
- why mel scale not bark etc.
go through loss network and explain weights (quality, size, wetness, odg) etc. make links to how data was processed for this task

- cite similar papers in zotero loss subcollection (like LEAN, etc.) for fast audio classification
  - why our loss model was based on CNN14
  - runtime (inference) evaluation

- general comparison of different loss functions in audio ML (sisnr, pesq, mse, l1, our own)


- plot is little pointless here: akin to plotting wetness and size against theirselfs, BUT in the end this quality function will be estimated using Neural Network


LOSS Net is based on CNN14 as shown in PANNs paper. Originally for near real time audio tagging => made sense to use here.


// ── Main backbone ──────────────────────────────────────────────────────────
#draw-network(
  (
    // ── Input ──────────────────────────────────────────────────────────────
    (
      type: "input",
      image: none,
      height: 8,
      depth: 1,
      label: "Audio",
      name: "audio",
    ),
    // ── MelSpectrogram ─────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.5,
      height: 8,
      depth: 2,
      fill: rgb("#9B59B6"),
      opacity: 0.85,
      label: "MelSpec",
      name: "mel",
      offset: 1.5,
      legend: "Feature Extractor (frozen)",
    ),
    // ── BN0 ────────────────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.3,
      height: 8,
      depth: 5,
      fill: rgb("#F39C12"),
      opacity: 0.85,
      label: "BN0",
      name: "bn0",
      offset: 1.5,
      legend: "Batch Normalization",
    ),
    // ── Conv Block 1 ───────────────────────────────────────────────────────
    (
      type: "conv",
      widths: (0.4, 0.4),
      height: 7,
      depth: 7,
      channels: (1, 64),
      label: "CB1\n1→64",
      name: "cb1",
      offset: 1.8,
      show-relu: true,
    ),
    (
      type: "pool",
      height: 3.5,
      depth: 3.5,
      label: "Avg\n2×2",
      name: "pool1",
    ),
    // ── Conv Block 2 ───────────────────────────────────────────────────────
    (
      type: "conv",
      widths: (0.4, 0.4),
      height: 6,
      depth: 6,
      channels: (64, 128),
      label: "CB2\n64→128",
      name: "cb2",
      offset: 1.5,
      show-relu: true,
    ),
    (
      type: "pool",
      height: 3,
      depth: 3,
      label: "Avg\n2×2",
      name: "pool2",
    ),
    // ── Conv Block 3 ───────────────────────────────────────────────────────
    (
      type: "conv",
      widths: (0.4, 0.4),
      height: 5,
      depth: 5,
      channels: (128, 256),
      label: "CB3\n128→256",
      name: "cb3",
      offset: 1.5,
      show-relu: true,
    ),
    (
      type: "pool",
      height: 2.5,
      depth: 2.5,
      label: "Avg\n2×2",
      name: "pool3",
    ),
    // ── Conv Block 4 ───────────────────────────────────────────────────────
    (
      type: "conv",
      widths: (0.4, 0.4),
      height: 4,
      depth: 4,
      channels: (256, 512),
      label: "CB4\n256→512",
      name: "cb4",
      offset: 1.5,
      show-relu: true,
    ),
    (
      type: "pool",
      height: 2,
      depth: 2,
      label: "Avg\n2×2",
      name: "pool4",
    ),
    // ── Conv Block 5 ───────────────────────────────────────────────────────
    (
      type: "conv",
      widths: (0.4, 0.4),
      height: 3,
      depth: 3,
      channels: (512, 1024),
      label: "CB5\n512→1024",
      name: "cb5",
      offset: 1.5,
      show-relu: true,
    ),
    (
      type: "pool",
      height: 1.5,
      depth: 1.5,
      label: "Avg\n2×2",
      name: "pool5",
    ),
    // ── Global Pooling ─────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.25,
      height: 5,
      depth: 0,
      fill: rgb("#1ABC9C"),
      opacity: 0.85,
      label: "GPool\n1024",
      name: "gpool",
      offset: 2,
      legend: "Global Pooling",
    ),
    // ── FC Shared ──────────────────────────────────────────────────────────
    (
      type: "fc",
      height: 4,
      depth: 0,
      channels: (512,),
      label: "FC\n512",
      name: "fc_shared",
      offset: 2,
    ),
    // ── ODG Head ───────────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.25,
      height: 3.5,
      depth: 0,
      fill: rgb("#E74C3C"),
      opacity: 0.85,
      label: "ODG\n128→1",
      name: "odg",
      offset: 3,
      legend: "Task Head",
    ),
    // ── Size Head ──────────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.25,
      height: 2.5,
      depth: 0,
      fill: rgb("#C0392B"),
      opacity: 0.85,
      label: "Size\n64→1",
      name: "size",
      offset: 1.5,
      show-connection: false,
      legend: "Task Head (size/wet)",
    ),
    // ── Wetness Head ───────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.25,
      height: 2.5,
      depth: 0,
      fill: rgb("#C0392B"),
      opacity: 0.85,
      label: "Wet\n64→1",
      name: "wetness",
      offset: 1.5,
      show-connection: false,
    ),
    // ── Quality Head ───────────────────────────────────────────────────────
    (
      type: "custom",
      width: 0.35,
      height: 4,
      depth: 0,
      fill: rgb("#27AE60"),
      opacity: 0.9,
      label: "Quality\n515→128→1",
      name: "quality",
      offset: 4,
      legend: "Quality Output",
    ),
  ),

  connections: (
    (from: "fc_shared", to: "size", type: "skip", mode: "air", label: "512", pos: 4),
    (from: "fc_shared", to: "wetness", type: "skip", mode: "air", label: "512", pos: 5),
    (from: "odg", to: "quality", type: "skip", mode: "air", label: "cat+3", pos: 5),
    (from: "size", to: "quality", type: "skip", mode: "air", pos: 4),
    (from: "wetness", to: "quality", type: "skip", mode: "air", pos: 3),
    (from: "fc_shared", to: "quality", type: "skip", mode: "depth", label: "512", pos: 7),
  ),

  palette: "warm",
  show-legend: true,
  legend-title: "Layer Types",
  show-relu: true,
  scale: 60%,
  stroke-thickness: 1,
  depth-multiplier: 0.22,
)


#draw-network(
  (
    (
      type: "input",
      image: none,
      height: 5,
      depth: 5,
      label: "Input\n(B,C_in,H,W)",
      name: "in",
    ),
    (
      type: "conv",
      widths: (0.5,),
      height: 5,
      depth: 5,
      channels: (3, "C_out"),
      label: "Conv2d 3×3\nstride=1 pad=1",
      name: "c1",
      offset: 2,
      show-relu: true,
    ),
    (
      type: "custom",
      width: 0.3,
      height: 5,
      depth: 5,
      fill: rgb("#F39C12"),
      opacity: 0.85,
      label: "BN + ReLU",
      name: "bn1",
      offset: 1,
      legend: "BN + ReLU",
    ),
    (
      type: "conv",
      widths: (0.5,),
      height: 5,
      depth: 5,
      channels: ("C_out", "C_out"),
      label: "Conv2d 3×3\nstride=1 pad=1",
      name: "c2",
      offset: 2,
      show-relu: true,
    ),
    (
      type: "custom",
      width: 0.3,
      height: 5,
      depth: 5,
      fill: rgb("#F39C12"),
      opacity: 0.85,
      label: "BN + ReLU",
      name: "bn2",
      offset: 1,
    ),
    (
      type: "pool",
      height: 2.5,
      depth: 2.5,
      label: "AvgPool\n(pool_size)",
      name: "pool",
    ),
    (
      type: "input",
      image: none,
      height: 2.5,
      depth: 2.5,
      label: "Output\n(B,C_out,H/2,W/2)",
      name: "out",
      offset: 1,
    ),
  ),
  connections: (),
  palette: "warm",
  show-legend: true,
  legend-title: "Sub-layers",
  show-relu: true,
  scale: 90%,
)


#figure(
  caption: [],
  draw-network(
    (
      (type: "input", height: 8, depth: 1, label: "Input", name: "img"),
      (
        type: "conv",
        // width: 1,
        height: 6,
        depth: 1,
        label: "Conv1",
        name: "c1",
      ),
      (type: "conv", height: 6, depth: 1, label: "Dilated Causal Conv", name: "c1"),
      (type: "conv", height: 6, depth: 1, label: "Weight Normalization", name: "c1"),
      (type: "conv", height: 6, depth: 1, label: "Dropout", name: "c1"),
      (type: "conv", height: 6, depth: 1, label: "Relu", name: "c1"),
      // Upsampling path
      (
        type: "conv",
        channels: ("K", "I/32"),
        widths: (0.3,),
        height: 2.5,
        depth: 2.5,
        label: "fc8 to conv",
        name: "s32",
        offset: 0.8,
        show-relu: false,
      ),
      (type: "deconv", channels: ("K", "I/16"), height: 3.5, depth: 3.5, name: "up1", offset: 1),
      (type: "sum", radius: 0.5, symbol: "+", name: "add1", offset: 1),
      (type: "deconv", height: 5, depth: 5, channels: ("K", "I/8"), name: "up2", offset: 0.5),
      (type: "sum", radius: 0.5, symbol: "+", name: "add2", offset: 1),
      (type: "deconv", height: 8, depth: 8, channels: ("K",), label: "Deconv", name: "up3", offset: 0.5),
      (type: "convsoftmax", height: 8, depth: 8, channels: ("K", "I"), label: "softmax", offset: 1),
    ),
    connections: (
      (
        from: "c1",
        to: "add1",
        type: "skip",
        mode: "flat",
        pos: 3,
        // layers: (
        //   (type: "conv", channels: ("K", "I/16"), widths: (0.3,), height: 2, depth: 3.5, name: "s16", show-relu: false),
        // ),
      ),
      (
        from: "p3",
        to: "add2",
        type: "skip",
        mode: "flat",
        pos: 6,
        layers: (
          (type: "conv", channels: ("K", "I/8"), widths: (0.3,), height: 2, depth: 3.5, name: "s16", show-relu: false),
        ),
      ),
    ),
    palette: "warm",
    show-legend: true,
    legend-title: "Layers",
    scale: 50%,
    stroke-thickness: 1,
    depth-multiplier: 0.3,
    show-relu: true,
  ),
)

== Dereverberation Network<impl_derev_net>
- it was shown that modifying the Conv TasNet TCN based architecture for a fully generative approach (no mask, but generate the final audio from the TCN representation) is not feasable with low computational cost (overfittable but doesn't generalize well)
  - show plots


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
