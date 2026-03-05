#import "/thesis/utils/todo.typ": TODO
#import "@preview/statastic:1.0.0": arrayAvg, arrayStd
#import "@preview/lilaq:0.5.0" as lq

= Results

#figure(
  caption: [],
  image("/experiments/perceptual-quality/plots/epoch_195-odg-perceptual_net_best.svg"),
)



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


#let v(array) = [#calc.round(arrayAvg(array), digits: 2) #sym.plus.minus  #calc.round(arrayStd(array), digits: 2)]

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
      [*MSE*], v(stormCSV.mse), v(convtasnetCSV.mse),
      [*SI-SNR*], v(stormCSV.si_snr), v(convtasnetCSV.si_snr),
      [*PESQ-WB*], v(stormCSV.pesq_wb), v(convtasnetCSV.pesq_wb),
      [*PESQ-NB*], v(stormCSV.pesq_nb), v(convtasnetCSV.pesq_nb),
      [*ODG*], v(stormCSV.odg), v(convtasnetCSV.odg),
      [*DI*], v(stormCSV.di), v(convtasnetCSV.di),
      [*Runtime*], [6:14:51], [0:04:15],
    )
  },
)

#let d(a, b) = {
  lq.diagram(
    xaxis: (
      ticks: range(1, 3).zip(([StoRM], [Conv-TasNet])),
      subticks: none,
    ),
    ylabel: "MSE",

    lq.boxplot(a, outliers: none, median: rgb(171, 105, 144)),
    lq.boxplot(b, outliers: none, x: 2, median: rgb(171, 105, 144)),
  )
}

#figure(
  caption: [
    Boxplot comparison of the MSE metric for the evaluation of dereverberation performance of diverse audio samples.
  ],
  lq.diagram(
    xaxis: (
      ticks: range(1, 3).zip(([StoRM], [Conv-TasNet])),
      subticks: none,
    ),
    ylabel: "MSE",

    lq.boxplot(stormCSV.mse, outliers: none, median: rgb(171, 105, 144)),
    lq.boxplot(convtasnetCSV.mse, outliers: none, x: 2, median: rgb(171, 105, 144)),
  ),
)

#figure(
  caption: [],
  d(stormCSV.si_snr, convtasnetCSV.si_snr)
)
