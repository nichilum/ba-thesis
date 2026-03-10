#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ

= Implementation & Experimental Setup

== Conv-TasNet for diverse audio dereverberation
- no proper weights for training available
  - I forgot the proper reasoning why we did not use weights available on huggingsface and google drive (linked on one github repo)
    - only for speaker seperation
    - reached out to original authors but did not get a response
- used model implementation by the author linked in the original paper @luoConvTasNetSurpassingIdeal2019
  - trained using LibriMix dataset @panayotovLibrispeechASRCorpus2015, original WSJ0-2mix and WSJ0-3mix datasets @garofolojohns.CSRIWSJ0Complete2007 are not publically available
  - original loss function (SI-SNR) only resulted in no convergence (stayed negative) and thus unusable results
  - switched SI-SNR loss to MSE loss which resulted in convergence and usable results (propably some oversight on our side)
  - show training and validation loss plots
  - show some example predictions (spectrograms and audio) in results?!

#import "@preview/neural-netz:0.3.0": draw-network

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

== Own Implementation
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
