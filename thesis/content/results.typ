#import "/thesis/utils/todo.typ": TODO
#import "@preview/statastic:1.0.0": arrayAvg, arrayStd
#import "@preview/lilaq:0.5.0" as lq
#import "/thesis/utils/diagram.typ": diagram
#import "/thesis/utils/author.typ": *

#let d(storm, tasnet, ylabel) = {
  lq.diagram(
    xaxis: (
      ticks: range(1, 3).zip(([StoRM], [Conv-TasNet])),
      subticks: none,
    ),
    ylabel: ylabel,

    // lq.boxplot(storm, outliers: ".", median: rgb(171, 105, 144)),
    // lq.boxplot(tasnet, outliers: ".", x: 2, median: rgb(171, 105, 144)),
    lq.boxplot(storm, outliers: none, median: rgb(171, 105, 144)),
    lq.boxplot(tasnet, outliers: none, x: 2, median: rgb(171, 105, 144)),
  )
}

#let loadAnalysisCSV(filename) = {
  let input = csv(filename)
  (
    "ids": input.at(0).slice(1),
    "mse": input.at(1).slice(1).map(d => float(d)).sorted(key: it => it),
    "si_snr": input.at(2).slice(1).map(d => float(d)).sorted(key: it => it),
    "pesq_wb": input.at(3).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "pesq_nb": input.at(4).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "odg": input.at(5).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "di": input.at(6).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
  )
}


#let v(array, digits: 2, std_digits: 2) = [#calc.round(arrayAvg(array), digits: digits) #sym.plus.minus  #calc.round(
    arrayStd(array),
    digits: std_digits,
  )]

#let stormCSV = loadAnalysisCSV("../data/export20260204-125026.csv")
#let convtasnetCSV = loadAnalysisCSV("../data/export20260205-120712.csv")

= Results<results>

== Conv-TasNet
#leo
As explained in @impl_conv_tasnet, we trained a Conv-TasNet model from scratch on the LibriSpeech `train-clean-100` split, using the original Conv-TasNet architecture @luoConvTasNetSurpassingIdeal2019 and training procedure, but with @MSE instead of @SI-SNR as the loss function.

Three loss functions were evaluated in total. The @SI-SNR, which serves as the original Conv-TasNet training objective, did not converge: the loss remained negative throughout and continued to decrease without producing usable predictions, with the best checkpoint reaching a validation @SI-SNR of $-69.72$ dB after 118 epochs (cf. @conv_tasnet_loss_comparison). A @MSS likewise showed convergence but only at an unreasonably high value, reaching a validation loss of $165,861.84$ after 119 epochs (cf. @conv_tasnet_mss_loss). This could be attributed to some configuration choices that were set implicitly and thus often fail to provide informative gradients as claimed by #cite(<schwarMultiScaleSpectralLoss2023>, form: "prose", style: "chicago-author-date").

Switching to a standard @MSE loss resolved the issue: training converged stably to a validation loss of approximately 0.0009 after 125 epochs (cf. @conv_tasnet_loss_comparison).

#TODO[we do not know the reason for this, might be a user error]

#diagram(
  caption: [
    Training curves for Conv-TasNet with different loss functions. Smoothed using an exponential moving average with $alpha=0.05$.
  ],
  short-caption: [Training curves for Conv-TasNet with different loss functions],
  image("../figures/conv_tasnet_loss_comparison.svg"),
)<conv_tasnet_loss_comparison>

#diagram(
  caption: [
    Training curve for Conv-TasNet with @MSS loss. Smoothed using an exponential moving average with $alpha=0.05$.
  ],
  short-caption: [Training curve for Conv-TasNet with @MSS:short loss],
  image("../figures/conv_tasnet_mss_loss.svg"),
)<conv_tasnet_mss_loss>

The @MSE\-trained model was evaluated on the LibriSpeech `test-clean` split as well as on a diverse random subset taken from AudioSet covering speech, music, vehicles, and environmental sounds.

On speech samples the model reduces reverberation tails and produces audible dereverberation. It can also be observed that the model applies a low-pass filter, reducing high frequencies above about 2.5 kHz by about 20 dB (see @spectrogram_comparison).

Detailed results of the evaluation done on the @MSE\-trained model can be seen in @conv_tasnet_metrics. A positive @SI-SNR and @SI-SDR indicate dereverberation, while the @PESQ score still indicates a _"poor"_ performance (cf. @fun_pesq)

#TODO[write about @conv_tasnet_metrics, and how it may show signs of dereverberation (or it doesnt, but its purely audible)]

#TODO[

  explain that the MSE checkpoint does return comparable SISNR values from the librispeech dataset as the original ConvTasNet paper reports

  15.3 dB in @luoConvTasNetSurpassingIdeal2019 vs 10.0 dB in @conv_tasnet_metrics

]

#diagram(
  caption: [
    Spectrogram comparison of input (left) and output (middle) of the @MSE\-trained Conv-TasNet on a speech (top) and music (bottom) sample.
  ],
  short-caption: [Spectrogram comparison of input and output of the @MSE\-trained Conv-TasNet],
  image("../figures/spectrogram_comparison.png"),
)<spectrogram_comparison>

On music and non-speech content, however, the model introduces noticeable timbral artifacts.

#diagram(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),
    [SI-SNR], [10.036], [3.856], [-6.669], [19.226],
    [SI-SDR], [10.003], [3.872], [-10.015], [19.220],
    [PESQ], [1.726], [0.304], [1.117], [3.015],
    [WV-MOS], [1.695], [0.441], [1.229], [3.350],
  ),
  caption: [Conv-TasNet dereverberation metrics on LibriSpeech `test-clean` (N = 2620)],
)<conv_tasnet_metrics>

These limitations -- the 4 kHz bandwidth ceiling, speech-only training data, and the uncertainty of @SI-SNR as a training objective for diverse audio -- motivate the development of a dedicated dereverberation model trained on broadband diverse content and supported by a perceptual loss network.

== StoRM
#leo
Unlike Conv-TasNet, StoRM was not trained from scratch. We used the official pretrained dereverberation checkpoint provided by the authors, trained on the WSJ0 corpus reverberated with the REVERB challenge dataset @lemercierStoRMDiffusionbasedStochastic2023 @kinoshitaReverbChallengeCommon2013 @garofolojohns.CSRIWSJ0Complete2007. The training data consists of speech recordings sampled at 16 kHz, establishing an 8 kHz frequency ceiling. Architecturally, StoRM follows a generative stochastic regeneration approach: a discriminative denoiser first produces an initial estimate of the clean signal, which a score-based diffusion model then refines through a learned reverse process @lemercierStoRMDiffusionbasedStochastic2023. This contrasts with Conv-TasNet's discriminative masking, and the iterative inference required by the diffusion component has direct implications for computational cost.

On in-domain speech signals, StoRM achieves strong dereverberation quality. @storm_paper_metrics shows the evaluation from the original paper. These numbers serve as an upper bound for speech dereverberation quality achievable with this model.

#diagram(
  caption: [StoRM dereverberation metrics on the WSJ0+REVERB test set, reproduced from @lemercierStoRMDiffusionbasedStochastic2023.],
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    table.header([*Method*], [*WV-MOS*], [*PESQ*], [*ESTOI*], [*SI-SDR*], [*SI-SIR*], [*SI-SAR*]),
    [Mixture],
    [$1.78 plus.minus 0.99$],
    [$1.36 plus.minus 0.19$],
    [$0.46 plus.minus 0.12$],
    [$-7.3 plus.minus 5.5$],
    [$-7.5 plus.minus 5.4$],
    [---],

    [SGMSE+],
    [$3.49 plus.minus 0.39$],
    [$2.66 plus.minus 0.45$],
    [$0.85 plus.minus 0.06$],
    [$2.4 plus.minus 7.2$],
    [$11.6 plus.minus 9.9$],
    [$2.8 plus.minus 6.8$],

    [NCSN++],
    [$2.99 plus.minus 0.38$],
    [$2.08 plus.minus 0.47$],
    [$0.85 plus.minus 0.06$],
    [$6.1 plus.minus 3.8$],
    [$21.4 plus.minus 7.0$],
    [$6.1 plus.minus 3.7$],

    [GaGNet],
    [$2.40 plus.minus 0.52$],
    [$1.59 plus.minus 0.37$],
    [$0.68 plus.minus 0.09$],
    [$-0.5 plus.minus 4.8$],
    [$7.7 plus.minus 4.0$],
    [$0.2 plus.minus 5.1$],

    [*StoRM*],
    [$bold(3.73 plus.minus 0.32)$],
    [$bold(2.83 plus.minus 0.42)$],
    [$bold(0.88 plus.minus 0.04)$],
    [$bold(6.5 plus.minus 4.0)$],
    [$bold(22.9 plus.minus 8.2)$],
    [$bold(6.5 plus.minus 3.9)$],
  ),
)<storm_paper_metrics> In our own listening tests on speech samples, this quality is confirmed: reverberation tails are cleanly removed with rarely any audible artifacts (see @spectrogram_comparison_storm). Informally, the model appears to perform slightly worse on female voices, which may be attributable to a gender bias in the WSJ0 training corpus toward male utterances, even though the authors claim: "[...] about half the speakers are male and half female" @garofolojohns.CSRIWSJ0Complete2007. Compared to Conv-TasNet, StoRM produces a markedly wider frequency response up to 8 kHz, avoiding the strong low-pass filtering effect observed in the MSE-trained Conv-TasNet output.

#diagram(
  caption: [
    Spectrogram comparison of input (left) and output (middle) of StoRM on a speech (top) and music (bottom) sample.
  ],
  short-caption: [Spectrogram comparison of input and output of StoRM],
  image("../figures/spectrogram_comparison_storm.png"),
)<spectrogram_comparison_storm>

On music and other non-speech content, the model's behaviour is less predictable. As the pretrained checkpoint has no exposure to non-speech signals during training, generalisation is limited to the extent that spectral patterns of broadband audio are covered by the speech-domain prior. In listening tests, music samples processed by StoRM tend to exhibit subtle timbral changes compared to the unprocessed input, without achieving a consistent reduction of the reverberant tail. This out-of-domain degradation is expected given the training data composition and is explored further in the quantitative comparison in the next section.

The iterative reverse diffusion inference requires many sequential neural network evaluations per sample, making StoRM substantially more expensive than Conv-TasNet. On a single H100 GPU (CLAIX-2023-ML), processing 2048 AudioSet samples took 6 h 14 m 51 s, compared to 4 m 15 s for Conv-TasNet --- approximately $88times$ slower (cf. @conv_tasnet_storm_comparison). Real-time application of this pretrained model is therefore not feasible without architectural modifications such as reducing the number of reverse diffusion steps or distillation.

== Perceptual Quality Network<results_percep_quality_net>
#jojo
- all results based on the dataset in @data_collection
- no results from initial perceptual quality network only cnn14 is discussed as
  - as @eval_percep_qual_net_init shows a direct improvement of cnn14 over initial implementation

#diagram(
  caption: [],
  image("/experiments/perceptual-quality/plots/epoch_195-odg-perceptual_net_best.svg"),
)

== Objective Quality Network<results_objective_quality_net>
#jojo
update score:
#diagram(
  caption: [],
  image("/experiments/perceptual-quality/plots/epoch_61-quality-perceptual_net_best.svg"),
)
update score and update loss:
#diagram(
  caption: [],
  image("/experiments/perceptual-quality/plots/epoch_166-quality-perceptual_net_best.svg"),
)

// this needs more thoughts
TOTAL NUMBER OF SAMPLES: 2896664400
Total length of audio coded in 44.1 kHz is 18.2455555556 hours
TOTAL LENGTH OF INFERENCE TIME: 7.738234307016683

== Dereverberation Network
#leo
#TODO[which versions do the want to show the results of? only the best one (then compare si-snr with perceptual?)]

#diagram(
  caption: [],
  image(
    "../../experiments/perceptual-quality/test_output/derev_tcn_v4_SISNR-epoch=16-val_loss=-14.3848/spectrograms/353-128309-0032.png",
  ),
)<derev_tcn_v4_sisnr>
#diagram(
  caption: [],
  image(
    "../../experiments/perceptual-quality/test_output/derev_tcn_v4_SISNR-epoch=55-val_loss=-16.6097/spectrograms/353-128309-0032.png",
  ),
)<derev_tcn_v4_sisnr_updated>

- one can see the effect of the additional #{55-16} epochs of training when comparing @derev_tcn_v4_sisnr and @derev_tcn_v4_sisnr_updated. While the reverberation tail is only slightly further reduced in magnitude, the high-frequency noise is substantially reduced, but still present. This is reflected in the SI-SNR improvement, which increases from $11.8$ dB to $16.2$ dB for this example.

- gating effect
- adds highs in some examples
  - makes sense because of "learning" at 44.1 kHz (@preprocessing_reverberation)
  - "upsampling" effect almost pleasant in some cases, but also adds unwanted artifacts in others
  - discussion: maybe reverberating at 44.1 kHz would have made sense (@disc_upsampling)
- if the input signal is not so reverberant, the output only gets dereverberated very little
- quality of dereverberation is highly dependent on the quality of the input signal

