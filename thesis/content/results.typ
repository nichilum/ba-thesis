#import "/thesis/utils/todo.typ": TODO
#import "@preview/statastic:1.0.0": arrayAvg, arrayStd
#import "@preview/lilaq:0.5.0" as lq

= Results

== Conv-TasNet

=== Loss Function

Three loss functions were evaluated. The @SI-SNR, which serves as the original Conv-TasNet training objective, did not converge: the loss remained negative throughout and continued to decrease without producing usable predictions, with the best checkpoint reaching a validation @SI-SNR of $-69.72$ dB after 118 epochs. A multi-scale spectral loss likewise showed convergence but only at a unreasonably high value, reaching a validation loss of $165,861.84$ after 119 epochs. Switching to a standard @MSE loss resolved the issue: training converged stably to a validation loss of approximately 0.0009 after 125 epochs.

//TODO: we do not know the reason for this, might be a user error

#figure(
  caption: [
    Training curves for Conv-TasNet with different loss functions. The SI-SNR loss did not converge to a positive value, while the MSE loss converged stably.
  ],
  image("../figures/conv_tasnet_loss_comparison.svg")
)

#figure(
  caption: [
    Training curve for Conv-TasNet with MSS loss. The loss converged but to an unreasonably high value, which did not produce usable predictions.
  ],
  image("../figures/conv_tasnet_mss_loss.svg")
)

#TODO[other word than failure? maybe "instability"?]
These limitations -- the 4 kHz bandwidth ceiling, speech-only training data, and the failure of @SI-SNR as a training objective for diverse audio -- motivate the development of a dedicated dereverberation model trained on broadband diverse content and supported by a perceptual loss network.

#figure(
  caption: [],
  image("/experiments/perceptual-quality/plots/epoch_195-odg-perceptual_net_best.svg"),
)



#let loadAnalysisCSV(filename) = {
  let input = csv(filename)
  (
    "ids": input.at(0).slice(1),
    "mse": input.at(1).slice(1).map(d => float(d)).sorted(key: it => it),
    "si_snr": input.at(2).slice(1).map(d => float(d) * -1).sorted(key: it => it),
    "pesq_wb": input.at(3).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "pesq_nb": input.at(4).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "odg": input.at(5).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
    "di": input.at(6).slice(1).map(d => float(d)).filter(e => not e.is-nan()).sorted(key: it => it),
  )
}


#let v(array, digits: 2, std_digits: 2) = [#calc.round(arrayAvg(array), digits: digits) #sym.plus.minus  #calc.round(arrayStd(array), digits: std_digits)]

#let stormCSV = loadAnalysisCSV("../data/export20260204-125026.csv")
#let convtasnetCSV = loadAnalysisCSV("../data/export20260205-120712.csv")


#figure(
  caption: [
    Comparison of different metrics for the evaluation of dereverberation performance of diverse audio samples, evaluated on 2048 random AudioSet samples. The runtime is measured on a single H100 GPU (CLAIX-2023-ML) #footnote([https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/fbd107191cf14c4b8307f44f545cf68a/]) with 2048 Random AudioSet Samples.
  ],
  {
    // set text(size: 7pt)
    table(
      columns: 3,
      align: (left, center, center),
      [*Network*], [StoRM], [Conv-TasNet],
      [*MSE*], v(stormCSV.mse, digits: 5, std_digits: 3), v(convtasnetCSV.mse, digits: 5, std_digits: 3),
      [*SI-SNR*], v(stormCSV.si_snr), v(convtasnetCSV.si_snr),
      [*PESQ-WB*], v(stormCSV.pesq_wb), v(convtasnetCSV.pesq_wb),
      [*PESQ-NB*], v(stormCSV.pesq_nb), v(convtasnetCSV.pesq_nb),
      [*ODG*], v(stormCSV.odg), v(convtasnetCSV.odg),
      [*DI*], v(stormCSV.di), v(convtasnetCSV.di),
      [*Runtime*], [6:14:51], [0:04:15],
    )
  },
)

#TODO[Spectogram of our output (storm and convtasnet) for audioset samples]

- high std in some metrics, especially in SISNR and MSE
- overall low odg and di
- low-ish pesq score
- good SISNR score, but high std
- _compare metrics to storm and convtasnet paper metrics with speech-samples_

#let d(storm, tasnet, ylabel) = {
  lq.diagram(
    xaxis: (
      ticks: range(1, 3).zip(([StoRM], [Conv-TasNet])),
      subticks: none,
    ),
    ylabel: ylabel,

    lq.boxplot(storm, outliers: none, median: rgb(171, 105, 144)),
    lq.boxplot(tasnet, outliers: none, x: 2, median: rgb(171, 105, 144)),
  )
}

#figure(
  caption: [Boxplot comparison of different metrics for the evaluation of dereverberation performance of diverse audio samples. (Outliers not shown)],
  grid(
    columns: 2,
    column-gutter: 1cm,
    row-gutter: .5cm,
    align: right,
    d(stormCSV.mse, convtasnetCSV.mse, "MSE"),
    d(stormCSV.si_snr, convtasnetCSV.si_snr, "SI-SNR"),
    d(stormCSV.pesq_wb, convtasnetCSV.pesq_wb, "PESQ-WB"),
    d(stormCSV.pesq_nb, convtasnetCSV.pesq_nb, "PESQ-NB"),
    d(stormCSV.odg, convtasnetCSV.odg, "ODG"),
    d(stormCSV.di, convtasnetCSV.di, "DI"),
  )
)

#TODO[Think about outliers in boxplots (only show some?)]

#figure(
  caption: [],
  image("../../experiments/perceptual-quality/test_output/derev_tcn_v4_SISNR-epoch=16-val_loss=-14.3848/spectrograms/353-128309-0032.png")
)

- gating effect
- adds highs in some examples
  - makes sense because of "learning" at 44.1 kHz (@preprocessing_reverberation)
  - "upsampling" effect almost pleasant in some cases, but also adds unwanted artifacts in others
  - discussion: maybe reverberating at 44.1 kHz would have made sense (@disc_upsampling)
- if the input signal is not so reverberant, the output only gets dereverberated very little
- quality of dereverberation is highly dependent on the quality of the input signal