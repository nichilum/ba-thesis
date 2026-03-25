#import "/thesis/utils/author.typ": *
#import "/thesis/utils/todo.typ": TODO
#import "@preview/diagraph:0.3.6": *
#import "/thesis/utils/diagram.typ": diagram
#import "/thesis/utils/author.typ": *

= Theoretical Background

== Acoustics

=== Reverberation

Reverberation is the "hanging-on" of sound in a room after the exciting signal has been removed @everestMasterHandbookAcoustics1989. It occurs naturally through late reflections of sound as described in @fun_natual_reverb or artificially utilizing convolution (@fun_conv_reverb), delay networks (@fun_delay_reverb) or room simulations (@fun_room_reverb).

==== Natural Reverberation<fun_natual_reverb>

Reverberation is casued by late reflections ($>50-80$ ms) @kuhn-rahloffSchallRaumUnd2025 of sound that overlap with the direct sound. This natural phenomenon can be observed in every environment that allows for sound reflections. An example is given by #cite(<everestMasterHandbookAcoustics1989>, form: "prose", style: "chicago-author-date"):

@fun_growth_decay_sound (A) shows a spherical sound source S and a listener L. Once S is energized sound travels outward from S in all directions. Sound pressure at L instantly jumps to a value ($D$) that is less than that which left S due to spherical divergence and small losses in the air. Sound pressure continues to grow with each successive arrival of reflected components unit an equilibrium is reached (cf. @fun_growth_decay_sound (B)).

Once S is turned off the sound rays moving through the room lose their support and with each successive reflection they lose energy until they are considered dead. These reflections are also called the reverb tail. @fun_growth_decay_sound (C) shows the exponential decrease of the first reflection components.

While the growth of sound is perceived as almost instant the decay is slow.

#diagram(
  caption: [(A) Direct sound and exemplary first order reflections from source S arriving at listening position L. (B) The sound pressure at L grows stepwise. (C) The sound decays exponentially after the source ceases. Taken from #cite(<everestMasterHandbookAcoustics1989>, form: "prose", style: "chicago-author-date").],
  short-caption: [The growth and decay of sound in a room],
  image("/thesis/figures/sound_growth_and_decay.svg"),
)<fun_growth_decay_sound>

Natural reverberation is classified through the reverberation-time metric ($R T$). The $R T_60$ indicates the time is takes for sound to decay by 60 dB after the source stops.

==== Convolutional Reverberation<fun_conv_reverb>

The idea of convolving a perfect room impulse response has been formulated as early as 1970 by #cite(<schroederDigitalSimulationSound1969>, form: "prose", style: "chicago-author-date")

bla bla seit 2000 erst rechner schnell genug

convolutions are a fast operation in the frequency domain and on GPU devices @siddiqOptimizationConvolutionReverberation2020 @misicAnalysisCPUGPU2016


Hier formel:
$
  (f convolve g)(t) := integral_(-infinity)^infinity f(tau) g(t-tau) dif tau
$

da steht auch unnötig viel drin aber lowkey useless:
@valimakiFiftyYearsArtificial2012

==== Delay Networks<fun_delay_reverb>

@smithPhysicalAudioSignal2010

==== Room Simulations<fun_room_reverb>

@mannallRoomAcoustiCOpensourceRoom2025
@scheiblerPyroomacousticsPythonPackage2018
@allenImageMethodEfficiently1979
@vorlanderAuralizationFundamentalsAcoustics2008

=== Sound Quality<fun_sound_quality>

@fun_quality_metrics describes different metrics that are used to qualify how well a corrupted (later also called predicted, test or processed) audio signal sounds in comparison to its clean counterpart.
Multiple of these metrics are perceptually motivated. This section gives an overview over parameters that impact the perception of audio. It is important to note that not all parameters listed can or should be used by quality metrics to qualify how good audio sounds (cf. @analyze_loss_functions).

There are many different signal corruptions that impact perceived sound quality. Some occur naturally like reverberation or background noise, while others are of digital origin like undersampling, insufficient bit depth or paket loss.

The impact of different signal corruptions differs between application. Arguably the most important parameter in perceived speech quality is the amount of noise introduced in the signal as speech intelligibility degrades significantly with lower speech-to-noise ratios @longArchitecturalAcoustics2006. Speech intelligibility is also affected by excessive reverberation as lower level consonants are masked by the reverb tail. On the other hand, a completely dead room, or an outdoor environment, is not optimal for intelligibility as the direct sound level may be too low to clearly hear speech. Also affected by reverberation is music as different genres benefit from different $R T$s. Absent reverberation makes music sound thin and weak @everestMasterHandbookAcoustics1989.

The human auditory complex perceives signals in the 20 to 20 kHz range @isoAcousticsReferenceZero2019a @pumphreyUpperLimitFrequency1950 @mollerHearingLowInfrasonic2004. A digital representation of a signal should therfore be sampled at at least 40 kHz @shannonCommunicationPresenceNoise1949. While it is possible to represent speech using 8 kHz of bandwidth @itu-tG711PulseCode1988 music and other broadband signals would lack important high frequency informations.

While converting analog signals into digital representations bit depth is another important factor. An 8 bit quantization allows for $49.93 "dB"$ of dynamic range. While speech requires a dynamic range of about 40 to 50 dB @dunnStatisticalMeasurementsConversational1940 @pavlovicSIISpeechIntelligibility2018, music sometimes featues signal-to-noise ratios of up to 80 dB. Through the introduction of the compact disc a quantization of 16 bit (offering $98.09$ dB of dynamic range) has become the standard @frenzelAudioElectronics2010.

Most of the above named signal qualities also impact @STT performance of neural networks @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021. A signal that is badly reverberated and noisy is harder to transcribe than an anechoic one. While humans have the added benefit of being able to deduce what was said by filling the gaps between understood words @longArchitecturalAcoustics2006 machines generally lack this ability.

Percieved sound quality is not only guided by signal corruptions but also by the overall frequency makeup of the signal. Although annoyance due to a sound can be highly subjective it was shown that critical bands defined by the cochlea define frequency regions in which two simultaneously played sounds are perceived as dissonant @longArchitecturalAcoustics2006.

== Neural Networks<fun_neural_networks>
Neural networks are parameterized function approximators composed of interconnected layers of simple computational units. During training, their weights and biases are iteratively adjusted so that the network maps an input to a desired output through minimizing a loss function @goodfellowDeepLearning2016. In practice, modern architectures differ mainly in how they organize these computations and which inductive bias they impose on the data. For this thesis, the most relevant families are @CNN:pl, @RNN:pl, and @TCN:pl.

=== Architectures
#leo
==== @CNN:short<fun_cnn>

@CNN:pl are feed-forward neural networks that process structured inputs by applying learned convolution kernels over local neighborhoods. Instead of connecting every input element to every neuron, a convolutional layer reuses the same filter weights across the full input. This weight sharing reduces the number of parameters and makes the network particularly effective at detecting local patterns such as edges in images, harmonics in spectrograms, or short waveform structures @goodfellowDeepLearning2016. Stacking multiple convolution layers increases the receptive field, so deeper layers can combine simple local features into more abstract representations.

In audio machine learning, @CNN:pl are widely used on time-frequency representations because spectrograms preserve local correlations in time and frequency that can be captured efficiently by two-dimensional filters. One-dimensional variants are also common when operating directly on waveforms or learned feature sequences. Their main advantages are computational efficiency, parameter sharing, and good parallelizability on modern hardware. A limitation is that long-range temporal dependencies are not represented explicitly and must instead be captured through depth, pooling, or large receptive fields @grau-haroComprehensiveEvaluationCNNBased2025 @kongPANNsLargeScalePretrained2020.

==== @RNN:short<fun_rnn>

@RNN:pl are designed for sequential data. In addition to the current input, each recurrent step receives a hidden state that carries information from previous time steps, allowing the network to model temporal dependencies. This makes @RNN:pl conceptually well suited for signals, text, and time series, where the interpretation of one sample often depends on earlier context, @rumelhartLearningRepresentationsBackpropagating1986 @goodfellowDeepLearning2016. In audio applications, recurrent layers have been used for tasks such as speech enhancement, quality prediction, and sequence modeling because they can aggregate information over longer temporal spans than shallow feed-forward models.

#diagram(
  caption: [@RNN architecture for an input sequence $x$, hidden connections parametrized by a weight matrix $U$ and hidden-to-hidden recurrent connections parametrized by a weight matrix $W$. Shown on the right is the unrolled version of an @RNN cell across three time steps. Replicated from #cite(<goodfellowDeepLearning2016>, form: "prose", style: "chicago-author-date").],
  short-caption: [@RNN:short architecture and unfolded version],
  grid(
    columns: (1fr, 3fr),
    align: (left, right),
    raw-render(
      ```
      digraph RNN_Cell {
        rankdir=TB;
        splines=true;
        node[math=true];
        nodesep=0.4;
        ranksep=0.42;

        // Hidden state (cell)
        node [shape=rectangle, width=0.8, height=1.2];
        h [label="h"];

        // Input (red)
        node [shape=circle, width=0.6, height=0.6];
        x [label="x"];

        // Output (blue)
        node [shape=circle, width=0.6, height=0.6];
        y [label="hat(y)"];


        // Edges
        x -> h [label="U"];
        h -> y [label="V"];

        // Recurrent self-loop
        h -> h [label="W"];

        // Layout alignment
        { rank=same; x; }
        { rank=same; h; }
        { rank=same; y; }

      }
      ```,
      // width: 4cm,
      height: 10cm,
    ),
    raw-render(
      ```
      digraph RNN {
        rankdir=LR;
        splines=true;
        nodesep=0.3;
        ranksep=0.3;

        node[math=true];

        // Node styles
        node [shape=rectangle, width=0.6, height=1.2];
        edge [fontsize=10];

        // Hidden states (green blocks)
        h_prev [label="h^((t-1))"];
        h_t    [label="h^((t))"];
        h_next [label="h^((t+1))"];

        // Inputs (red circles)
        node [shape=circle, fillcolor="#F4A6A6", width=0.6, height=0.6];
        x_prev [label="x^((t-1))"];
        x_t    [label="x^((t))"];
        x_next [label="x^((t+1))"];

        // Outputs (blue circles)
        node [shape=circle, fillcolor="#A6C8FF", width=0.6, height=0.6];
        y_prev [label="hat(y)^((t-1))"];
        y_t    [label="hat(y)^((t))"];
        y_next [label="hat(y)^((t+1))"];

        // Edges: input to hidden (U)
        x_prev -> h_prev [label="U"];
        x_t    -> h_t    [label="U"];
        x_next -> h_next [label="U"];

        // Edges: hidden to hidden (W)
        h_prev -> h_t    [label="W"];
        h_t    -> h_next [label="W"];

        // Edges: hidden to output (V)
        h_prev -> y_prev [label="V"];
        h_t    -> y_t    [label="V"];
        h_next -> y_next [label="V"];

        // Rank alignment
        { rank=same; x_prev; h_prev; y_prev; }
        { rank=same; x_t;    h_t;    y_t; }
        { rank=same; x_next; h_next; y_next; }
      }
      ```,
      // width: 10cm,
      height: 10cm,
    ),
  ),
)



Classical @RNN:pl suffer from vanishing and exploding gradients when the dependency horizon becomes long. For this reason, gated variants such as @LSTM @hochreiterLongShortTermMemory1997, @BLSTM and @GRU networks @choPropertiesNeuralMachine2014 are commonly used in practice. These architectures regulate which information is stored, updated, or forgotten, improving training stability and long-term memory. Their main drawback is that recurrent processing is inherently sequential, which limits parallelization during training and inference compared to purely convolutional models @fuQualityNetEndtoEndNonintrusive2018 @defossezMusicSourceSeparation2019.

==== @TCN:short<fun_tcn>

@TCN:pl adapt the convolutional idea to sequence modeling by applying one-dimensional convolutions along the temporal axis. To cover long contexts efficiently, they often use dilated convolutions, where filter taps are spaced apart by increasing dilation factors. This enlarges the receptive field without requiring very deep networks or large kernels. Depending on the application, @TCN:pl can be implemented causally, where each output depends only on the present and past, or non-causally, where future context is also available @baiEmpiricalEvaluationGeneric2018.

#import "@preview/fletcher:0.5.8": diagram as fletcher-diagram, edge, node

#let block(pos, label, tint) = node(
  pos,
  align(center, label),
  width: 50mm,
  height: 8mm,
  fill: tint.lighten(60%),
  stroke: 1pt + tint.darken(20%),
  corner-radius: 4pt,
)

#let green = rgb("8FBF8F")
#let yellow = rgb("D8C27A")

#diagram(
  caption: [TCN residual block, where the 1$times$1 Convolution is only added when input and output differ in dimensions. Replicated from #cite(<baiEmpiricalEvaluationGeneric2018>, form: "prose", style: "chicago-author-date").],
  short-caption: [TCN residual block],
  scale(x: 100%, y: 100%, fletcher-diagram(
    spacing: 6pt,
    cell-size: (10mm, 10mm),
    edge-stroke: 1pt,
    edge-corner-radius: 6pt,

    // Main vertical stack
    edge((2, -1), "d", "-|>"),
    node((2, 0), circle(radius: 3pt, fill: black)),

    edge("ll,d", "-|>"),
    block((0, 1), [Dilated Causal Conv], green),
    edge(),
    block((0, 2), [WeightNorm], yellow),
    edge(),
    block((0, 3), [ReLU], green),
    edge(),
    block((0, 4), [Dropout], yellow),

    edge(),
    block((0, 5), [Dilated Causal Conv], green),
    edge(),
    block((0, 6), [WeightNorm], yellow),
    edge(),
    block((0, 7), [ReLU], green),
    edge(),
    block((0, 8), [Dropout], yellow),

    // Sum node
    edge("d,rr", "-|>"),
    node((2, 9), circle(radius: 5pt, fill: rgb("E07A6F"))[+]),

    // Residual branch (right side)
    block((2, 6), [1$times$1 Conv (optional)], green),

    // Connections
    edge((2, 0), (2, 3), "ddd", "-|>"),
    edge((2, 3), (2, 6), "ddd", "-|>"),
    edge((2, 9), "d", "-|>"),
    // edge((2,6), (0,8), "drr,u", "-|>"),
  )),
)


#let circ(pos, fill) = node(
  outset: 0pt,
  inset: 0pt,
  pos,
  circle(radius: 10pt, fill: fill, stroke: 1pt + black),
)



Compared to @RNN:pl, @TCN:pl retain the ability to model long temporal structure while remaining fully convolutional and therefore highly parallelizable. They also provide explicit control over receptive field size through kernel width, depth, and dilation schedule. This makes them attractive for audio tasks that require a compromise between temporal context and computational efficiency. In this thesis, @TCN:pl are particularly relevant because Conv-TasNet uses a temporal convolutional network to estimate masks over an encoded waveform representation @luoConvTasNetSurpassingIdeal2019. The basic principles of @TCN:pl therefore form part of the architectural foundation for the dereverberation models discussed later.

#diagram(
  caption: [
    A dilated causal convolution with dilation factors $d=1,2,4$. Replicated from #cite(<leePredictiveSkillConvolutional2021>, form: "prose", style: "chicago-author-date").
  ],
  short-caption: [Dilated causal convolution],
  scale(
    x: 80%,
    y: 80%,
    fletcher-diagram(
      spacing: 1cm,
      cell-size: (10mm, 10mm),
      edge-stroke: 1pt,

      // --- INPUT (blue) ---
      circ((0, 0), rgb("#4A90E2")),
      circ((1, 0), rgb("#4A90E2")),
      circ((2, 0), rgb("#4A90E2")),
      circ((3, 0), rgb("#4A90E2")),
      circ((4, 0), rgb("#4A90E2")),
      circ((5, 0), rgb("#4A90E2")),
      circ((6, 0), rgb("#4A90E2")),
      circ((7, 0), rgb("#4A90E2")),

      // --- HIDDEN d=1 ---
      circ((0, 1), white),
      circ((1, 1), rgb("#FF3B30")),
      circ((2, 1), white),
      circ((3, 1), rgb("#FF3B30")),
      circ((4, 1), white),
      circ((5, 1), rgb("#FF3B30")),
      circ((6, 1), white),
      circ((7, 1), rgb("#FF3B30")),

      // --- HIDDEN d=2 ---
      circ((0, 2), white),
      circ((1, 2), white),
      circ((2, 2), white),
      circ((3, 2), rgb("#FF3B30")),
      circ((4, 2), white),
      circ((5, 2), white),
      circ((6, 2), white),
      circ((7, 2), rgb("#FF3B30")),

      // --- OUTPUT ---
      circ((0, 3), white),
      circ((1, 3), white),
      circ((2, 3), white),
      circ((3, 3), white),
      circ((4, 3), white),
      circ((5, 3), white),
      circ((6, 3), white),
      circ((7, 3), rgb("#F8E71C")),

      // --- VERTICAL EDGES ---
      edge((0, 0), (0, 1), "--|>", stroke: gray),
      edge((1, 0), (1, 1), "-|>"),
      edge((2, 0), (2, 1), "--|>", stroke: gray),
      edge((3, 0), (3, 1), "-|>"),
      edge((4, 0), (4, 1), "--|>", stroke: gray),
      edge((5, 0), (5, 1), "-|>"),
      edge((6, 0), (6, 1), "--|>", stroke: gray),
      edge((7, 0), (7, 1), "-|>"),

      edge((0, 1), (0, 2), "--|>", stroke: gray),
      edge((1, 1), (1, 2), "--|>", stroke: gray),
      edge((2, 1), (2, 2), "--|>", stroke: gray),
      edge((3, 1), (3, 2), "-|>"),
      edge((4, 1), (4, 2), "--|>", stroke: gray),
      edge((5, 1), (5, 2), "--|>", stroke: gray),
      edge((6, 1), (6, 2), "--|>", stroke: gray),
      edge((7, 1), (7, 2), "-|>"),

      edge((0, 2), (0, 3), "--|>", stroke: gray),
      edge((1, 2), (1, 3), "--|>", stroke: gray),
      edge((2, 2), (2, 3), "--|>", stroke: gray),
      edge((3, 2), (3, 3), "--|>", stroke: gray),
      edge((4, 2), (4, 3), "--|>", stroke: gray),
      edge((5, 2), (5, 3), "--|>", stroke: gray),
      edge((6, 2), (6, 3), "--|>", stroke: gray),
      edge((7, 2), (7, 3), "-|>"),

      // --- DILATION = 1 ---
      edge((0, 0), (1, 1), "-|>"),
      edge((1, 0), (2, 1), "--|>", stroke: gray),
      edge((2, 0), (3, 1), "-|>"),
      edge((3, 0), (4, 1), "--|>", stroke: gray),
      edge((4, 0), (5, 1), "-|>"),
      edge((5, 0), (6, 1), "--|>", stroke: gray),
      edge((6, 0), (7, 1), "-|>"),

      // --- DILATION = 2 ---
      edge((0, 1), (2, 2), "--|>", stroke: gray),
      edge((1, 1), (3, 2), "-|>"),
      edge((2, 1), (4, 2), "--|>", stroke: gray),
      edge((3, 1), (5, 2), "--|>", stroke: gray),
      edge((4, 1), (6, 2), "--|>", stroke: gray),
      edge((5, 1), (7, 2), "-|>"),

      // --- DILATION = 4 ---
      edge((0, 2), (4, 3), "--|>", stroke: gray),
      edge((1, 2), (5, 3), "--|>", stroke: gray),
      edge((2, 2), (6, 3), "--|>", stroke: gray),
      edge((3, 2), (7, 3), "-|>"),

      node((8, 0), "Input"),
      node((8, 1), "Hidden"),
      node((8, 2), "Hidden"),
      node((8, 3), "Output"),

      node((7.5, 0.5), [$d=1$]),
      node((7.5, 1.5), [$d=2$]),
      node((7.5, 2.5), [$d=4$]),
    ),
  ),
)


==== Autoencoders

Autoencoders try to convert an input vector into a code vector of lower dimensionality and then reconstruct an approximation of the original input from this code vector @hintonAutoencodersMinimumDescription1993. This network can be split into two parts, an encoder function $ bold(h) = f(bold(x)) $ and a decoder $ bold(r) = g(bold(h)) $ generating a reconstruction $bold(r)$ of the original input $bold(x)$ through a hidden layer $bold(h)$ @goodfellowDeepLearning2016 @lecunModelesConnexionnistesLapprentissage1987.

Of particular interest for this thesis are @DAE:pl which are a type of regularized autoencoder. Instead of minimizing some loss function
$ L(bold(x), g(f(bold(x)))) $
, where $L$ is some loss function penalizing the difference between the input and the reconstruction, the denoising autoencoder minimizes
$ L(bold(x), g(f(tilde(bold(x))))) $
, where $tilde(bold(x))$ is a stochastically corrupted version of the input. This forces the network to learn a more robust representation of the data, as it must be able to undo the corruption from a noisy version @goodfellowDeepLearning2016. In the context of dereverberation, @DAE:pl can be used to learn a mapping from reverberant to clean speech by treating the reverberant signal as a corrupted version of the clean signal. The encoder learns to extract relevant features that are invariant to reverberation, while the decoder reconstructs the clean signal from these features.

#let circ(pos, text) = node(
  outset: 0pt,
  inset: 0pt,
  pos,
  circle(text, radius: 15pt, stroke: 1pt + black),
)

#diagram(
  caption: [
    @DAE training procedure. Replicated from #cite(<goodfellowDeepLearning2016>, form: "prose", style: "chicago-author-date").
  ],
  short-caption: [@DAE training procedure],
  scale(
    x: 80%,
    y: 80%,
    fletcher-diagram(
      spacing: 1cm,
      cell-size: (15mm, 15mm),
      edge-stroke: 1pt,

      circ((0.5, 0), [$bold(h)$]),
      edge("<|-", [$f$]),
      circ((0, 1), [$tilde(bold(x))$]),
      circ((1, 1), [$L$]),
      edge("<|-"),
      circ((0, 2), [$bold(x)$]),
      edge((0, 2), (0, 1), "-|>", [$C(tilde(bold(x))|bold(x))$], left),
      edge((0.5, 0), (1, 1), "-|>", [$g$]),
    ),
  ),
)<dae_training_fig>

@dae_training_fig shows the training procedure of a @DAE. $C(tilde(bold(x))|bold(x))$ is the corruption process which gives a distribution over corrupted versions $tilde(bold(x))$, given an original input $bold(x)$. The loss function $L$ is then minimized with respect to the parameters of the encoder $f$ and decoder $g$ from training pairs $(bold(x), tilde(bold(x)))$ @goodfellowDeepLearning2016.

#TODO[talk about skip connections]
#import "@preview/neural-netz:0.3.0": draw-network

#diagram(
  caption: [
    A U-Net architecture (FCN-8) #TODO[cite] with skip connections. The left side is the downsampling path where the input is processed through a series of convolutional and pooling layers, while the right side is the upsampling path where the feature maps are upsampled and combined with corresponding feature maps from the downsampling path through skip connections.
  ],
  short-caption: [],
  draw-network(
    scale: 43%,
    (
      (type: "input", image: "default", height: 8, depth: 8, label: "Input", name: "img"),
      (
        type: "conv",
        channels: ("64", "64", "I"),
        widths: (0.5, 0.5),
        height: 8,
        depth: 8,
        label: "Conv1",
        name: "c1",
        offset: 1.9,
      ),
      (type: "pool", height: 6.5, depth: 6.5, name: "p1"),
      (
        type: "conv",
        channels: ("128", "128", "I/2"),
        widths: (0.6, 0.6),
        height: 6.5,
        depth: 6.5,
        label: "Conv2",
        name: "c2",
      ),
      (type: "pool", height: 5, depth: 5, name: "p2"),
      (
        type: "conv",
        channels: ("256", "256", "256", "I/4"),
        widths: (0.7, 0.7, 0.7),
        height: 5,
        depth: 5,
        label: "Conv3",
        name: "c3",
      ),
      (type: "pool", height: 3.5, depth: 3.5, name: "p3"),
      (
        type: "conv",
        channels: ("512", "512", "512", "I/8"),
        widths: (0.8, 0.8, 0.8),
        height: 3.5,
        depth: 3.5,
        label: "Conv4",
        name: "c4",
      ),
      (type: "pool", height: 2.5, depth: 2.5, name: "p4"),
      (
        type: "conv",
        channels: ("512", "512", "512", "I/16"),
        widths: (0.8, 0.8, 0.8),
        height: 2.5,
        depth: 2.5,
        label: "Conv5",
        name: "c5",
      ),
      (type: "pool", height: 1.5, depth: 1.5, name: "p5"),
      (
        type: "conv",
        channels: ("4096", "4096"),
        widths: (1.5, 1.5),
        height: 1.5,
        depth: 1.5,
        label: "fc to conv",
        name: "fc",
      ),
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
        from: "p4",
        to: "add1",
        type: "skip",
        mode: "flat",
        pos: 3,
        layers: (
          (type: "conv", channels: ("K", "I/16"), widths: (0.3,), height: 2, depth: 3.5, name: "s16", show-relu: false),
        ),
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
    show-relu: true,
  ),
)




=== Supervised and Self-Supervised Learning<self_supervised_and_supervised_learning>
#leo

Training data can be given to a network in different ways. The most common way is to use supervised learning where each input data point is paired with a target output. The network learns to map inputs to their corresponding targets by minimizing a loss function that quantifies the error between the predicted output and the true target. We call this process *supervied learning*, because one can imagine a supervisor who shows the network the correct answer for each input and the network learns to mimic this behavior @goodfellowDeepLearning2016.

*Self-supervised* learning is a form of unsupervised learning where, instead of relying on external labels, the data itself is used to generate supervisory signals. This is often done by augmentation of the input data to formulate this signal. Self-supervised learning has been successfull in recent years, especially in labeled data scarce domains such aus audio processing, as shown by #cite(<baevskiWav2vec20Framework2020>, form: "prose", style: "chicago-author-date"). However, as our task is a supervised regression from reverberant to dry audio, we cannot classify our approach as self-supervised.

=== Training of a Neural Network
#jojo
@fun_neural_networks states neural networks are iteratively trained through minimizing a loss function.
The loss function is parameterized through an input-output function as well as the weights of the model.
Optimizing the loss function means optimizing the weights.

We can image a multidimensional error landscape formed by the weights. To traverse this error landscape into a local minimum we use the partial derivative of the loss function, also called gradient. This process coined gradient descent is discussed in @fun_gradient_descent.

This gradient was historically computed analytically (see @fun_loss_function). Modern multi-million parameter networks make this approach impossible. To aid the process modern networks use automatic differentiation based on the backpropagation method as introduced by #cite(<rumelhartLearningRepresentationsBackpropagating1986>, form: "prose", style: "chicago-author-date").

==== Loss Function<fun_loss_function>

A loss function, also called cost function, is a qualitative function that is used to objectively measure model performance by calculating the deviation of the model's prediction to their ground truth counterpart. This deviation is mapped onto a real number that intuitively represents some error.

To introduce the application of loss functions we want to discuss one of the eariest and simplest neural networks called Adaline @widrowAdaptiveAdalineNeuron1960. This single-layer neural network defines its input-output function as:

$ y(bold(x),bold(w)) = sum_(n=1)^N x_n w_n + b $

where $bold(x)$ is the input vector, $bold(w)$ the weight vector, $N$ the number of inputs, $b$ some bias and $y$ the model ouput. Assuming that $x_0 = 1$ and $w_0 = b$ the output is simplified to:

$ y(bold(x),bold(w)) = sum_(n=1)^N x_n w_n $

. Adaline uses the @LMS algorithm to define its loss, also called cost function:

$ C(d, y) = (d - y(bold(x),bold(w)))^2 $

where $d$ is the desired target. For analytical simplicity the loss function is often denoted as:

$
  C(d, y) & = 1/2 (y(bold(x),bold(w)) - d)^2 \
          & = 1/2 (x_1 w_1 + x_2 w_2 + ... + x_n w_n - d)^2
$<fun_loss_func_equ>

. The partial derivative, also called gradient, can be calculated analytically (here for the first weight) by deriving the input-output function:

$
  (partial C)/(partial w_1) & = 1/2 dot 2 dot (y(bold(x),bold(w)) - d) dot y(bold(x),bold(w))'_w_1 \
                            & =(y(bold(x),bold(w)) - d) dot x_1
$

. The learning rule implementing this partial derivative is denoted as:

$ bold(w) arrow.l bold(w) + eta (d - y(bold(x),bold(w))) bold(x) $<fun_apply_gradient_to_loss_eq>

where $eta$ is some factor called the learning rate. This update rule implements gradient descent for linear regression.
It should be noted that $y$ is quadratic in the above loss function. Therefore no local minima are offered and only a global minium is approached @amariBackpropagationStochasticGradient1993.

It can be concluded from the example above that analytical derivation of such loss functions becomes near impossible for complex input-output functions featuring non-linearities (activation functions) and millions of parameters. To solve this issue the backpropagation algorithm is used.

==== Backpropagation and Autograd<fun_backpropagation>

Training a neural network happens in two steps. Initially the input is run through each of the networks functions. Through this process, called forward propagation, the neural network makes its best guess about the correct output.

Once an input-output pair is computed the neural network calculates the gradient of the error function in regards to its guess by traversing backwards through its layers, collecting the derivatives of the error with respect to the parameters of the functions which are later used to change each weight. This operation is also known as gradient descent (see @fun_gradient_descent).

The following section will discuss backpropagation as introduced by #cite(<rumelhartLearningRepresentationsBackpropagating1986>, form: "prose", style: "chicago-author-date").

Expanding on the network example of @fun_loss_function we define our multi-layer network as having a leftmost layer of input units, any number of intermediate layers and a rightmost layer of output units. Connections within a layer or from right to left are forbidden, but connections can skip intermediate layers.
The states of the units in each layer are determined by applying equations @fun_b_s_e_1 and @fun_b_s_e_2

$ x_j = sum_i y_i w_(j i) $<fun_b_s_e_1>
$ y_j = 1/(1+e^(-x_j)) $<fun_b_s_e_2>

, also called the forward pass, where @fun_b_s_e_2 is the sigmoid function which today is often replace by the @ReLU activation function.

The total error $E$ is defined as (cf. @fun_loss_func_equ)

$ E = 1/2 sum_c sum_j (y_(j,c) - d_(j,c))^2 $<fun_b_total_loss>
where $c$ is an index over all input-output paris, $j$ is an index over output units, $y$ is the actual state of an output unit and $d$ is the desired state.
The backward pass starts by computing the parital derivative of $E$ in respect to $x_j$ for each output unit. Differentiating @fun_b_total_loss for a single input-output pair

$
  E & =1/2(y_j -d_j)^2 \
    & = 1/2 ((1/(1+e^(-x_j))) - d_j)^2
$

by applying the chain rule gives:

$
  (partial E)/(partial x_j) & =(partial E)/(partial y_j) dot (dif y_j)/(dif x_j) \
                            & = (partial E)/(partial y_j) dot y_j (1-y_j)
$

. This shows the affecting change is just a linear function of the states of the layer before making it "easy" @rumelhartLearningRepresentationsBackpropagating1986 to compute how the error will be affected by changing states in the intermediate layers. For a weight $w_(j i)$ the derivative is

$
  (partial E)/(partial w_(j i)) & = (partial E)/(partial x_(j)) dot y_i
$

The output of the $i$#super("th") unit taking into account all emerging connections results in

$ (partial E)/(partial y_i) = sum_j (partial E)/(partial x_j) dot w_(j i) $

. This shows how $(partial E)/(partial y)$ of the output layer can be computed when $(partial E)/(partial y_i)$ of the layer before is given. This procedure can therefore be repeated for each layer going backwards.

Historically these computations have been done manually by the researchers @baydinAutomaticDifferentiationMachine2015. This task is tedious and error prone. Here automatic differentiation algorithms are of assistance. Pytorch's autograd system stores all functional computations that create the neural network's guess in a directed acyclic graph "whose leaves are the input tensors and roots are the output tensors. By tracing this graph from roots to leaves, you can automatically compute the gradients using the chain rule"
@AutogradMechanicsPyTorch. It is important to note that this automatic process requires every function to respect the input data's need for a gradient. During computation gradient calculation can be accidentally disabled. This problem can occur when using another neural network as the loss function. This is further discussed in @impl_derev_net.




==== Gradient Descent<fun_gradient_descent>

In @fun_backpropagation is is discussed how partial derivatives of the error function $E$ can be calculated either manually or through the use of an automatic differentiation system.

Once $gradient E$ is calculated each weight can be adjusted so that the loss is further minimized (cf. @fun_apply_gradient_to_loss_eq). Through this process is called gradient descent a local minium is searched. Finding a global minimum is not necessary, as experts "suspect that, for sufficiently large neural networks, most local minima have a low cost function value, and that it is not important to find a true global minimum" @goodfellowDeepLearning2016.

#cite(<rumelhartLearningRepresentationsBackpropagating1986>, form: "prose", style: "chicago-author-date") introduce the simplest version of gradient descent as the accumulation of all gradients over all training examples and changing each weight by an amount proportional to the accumulated $(partial E)/(partial w)$. There are in fact improvements to this approach in the @SGD method which approximates the gradient of the entire dataset over a small subset of training examples also called minibatches. This lowers the computational cost of calculating a gradient over the entire dataset which is especially useful when dealing with large amounts of data. It is not guaranteed that the @SGD method arrives at a local minimum in a reasonable amount of time, but often a useful "low enough" loss is found @goodfellowDeepLearning2016.

@SGD is used during training of our perceptual quality network as well as the dereverberation network.

==== Taxonomy of Loss Functions<fun_taxonomy_loss>

@fun_loss_function and @fun_backpropagation made clear what impact a loss function can have on the training process of a neural network. Over the recent years many different loss functions for different problem sets have been envisioned each best suited for a specific input-output function with specific input-output data pairs @ciampiconiSurveyTaxonomyLoss2024.

#diagram(
  caption: [A taxonomy of loss functions taken from #cite(<ciampiconiSurveyTaxonomyLoss2024>, form: "prose", style: "chicago-author-date").],
  short-caption: [A taxonomy of loss functions],
  image("/thesis/figures/taxonomy.svg"),
)<taxonomy_fig>

@taxonomy_fig shows a map that identifies five major tasks for which loss functions can be designed, namely regression, classification, ranking as well as generative and energy based models. Optimization strategies for each task category are proposed including error-based, probabilistic and margin based loss functions.

The problem of dereverberation can be attributed to a regressive task. It is shown that for the problem of regression, error-based loss functions are applicable. The following section will discuss error-based metrics which can be utilized as loss functions. An analysis of their performance as such is discussed in @analyze_loss_functions.

== Quality Metrics<fun_quality_metrics>
#jojo
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

#diagram(caption: [Interpretation of the Pearson coefficient], table(
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

The @SI-SNR:long first introduced as the @SI-SDR in @rouxSDRHalfbakedWell2018 is defined as:

$ "SI-SNR" = 10 log_10 ((||a s||^2)/(||a s - hat(s)||^2)), "where" a = (hat(s)^T s)/(||s||^2) $

. The @SI-SNR and @SI-SDR have identical formulas but different interpretations of the corruption applied to the signal $hat(s)$ (noise or distortion). The @SI-SNR measures the level of noise in the predicted signal in a way that is invariant to amplitude scaling of the signals. It has been used successfully in dereverberation tasks @luoConvTasNetSurpassingIdeal2019 but while providing invariance to signal scaling it too does not conform to the perceived loudness of the human ear nor provide invariance to signal shifting.

It can also be mentioned that there are other variants, like Source-to-Artifact Ratio (SAR), Source-to-Interference Ratio (SIR), Source-to-Distortion Ratio (SDR) and Signal-to-Noise Ratio (SNR), each with Scale-Invariant (SI) forms, which are used in the field of source separation and speech enhancement, but are all inspired by the usual definition of the SNR @vincentPerformanceMeasurementBlind2006.

=== @PESQ:short<fun_pesq>

Answering the shortcoming of metrics like the @MSE and @SI-SNR, the @PESQ:both model (a successor to the @BSD and @PSQM models) is both invariant to signal scaling and shifting. In regards to its predecessors @PESQ produces more accurate scores when the signal is affected by coding distortion, packet loss, background noise, filtering or variable delay. The @PESQ score reflects speech quality on a continuous scale ranging from 1 to 5 (cf. @pesq_score_interp).

#diagram(
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

#diagram(
  caption: [Structure of @PESQ:both model taken from #cite(<rixPerceptualEvaluationSpeech2001>, form: "prose", style: "chicago-author-date").],
  short-caption: [Structure of @PESQ:short model],
  raw-render(
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
  ),
)<speech_quality_pipeline>


=== @PEAQ:short<fun_peaq>

The @PEAQ model is based on the @PAQM model and has been an ITU-R recommendation since 1999 @rixPerceptualEvaluationSpeech2001. In general it compares two time aligned signals, one processed and one original. Concurrent frames of each signal are transformed to a basilar membrane representation whose differences are further analyzed by a cognitive model @thiedePEAQITUStandard2000 (cf. @audio_quality_pipeline). The two offered metrics, namely the @ODG:both and @DI:both, are therefore not invariant to signal shifting but they conform to the human perception of sound loudness. The @ODG corresponds with the @SDG and indicates the audio quality of the tested signal on a continuous scale from -4 (very annoying impairment) to 0 (imperceptible impairment). The @DI is a quality indicator like the @ODG except for its higher sensitivity towards very low signal qualities @khalifehPerceptualEvaluationAudio2017.

#diagram(
  caption: [High-level representation of the @PEAQ:both model taken from #cite(<thiedePEAQITUStandard2000>, form: "prose", style: "chicago-author-date").],
  short-caption: [High-level representation of the @PEAQ:short model],
  raw-render(
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
  ),
)<audio_quality_pipeline>
