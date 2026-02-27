#import "/thesis/utils/todo.typ": TODO
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart, plot
#import "@preview/statastic:1.0.0": arrayAvg, arrayMedian, arrayPercentile, arrayStd

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
  table(
    columns: 8,
    align: center,
    table.header([Network], [*MSE*], [*SI-SNR*], [*PESQ-WB*], [*PESQ-NB*], [*ODG*], [*DI*], [*Runtime*]),
    [StoRM],
    v(stormCSV.mse),
    v(stormCSV.si_snr),
    v(stormCSV.pesq_wb),
    v(stormCSV.pesq_nb),
    v(stormCSV.odg),
    v(stormCSV.di),
    [6:14:51],

    [Conv-TasNet],
    v(convtasnetCSV.mse),
    v(convtasnetCSV.si_snr),
    v(convtasnetCSV.pesq_wb),
    v(convtasnetCSV.pesq_nb),
    v(convtasnetCSV.odg),
    v(convtasnetCSV.di),
    [0:04:15],
  ),
)


#cetz.canvas({
  plot.plot(
    size: (5, 5),
    x-tick-step: none,
    y-tick-step: 0.4,
    y-mode: "log",
    y-format: "sci",
    {
      plot.add-boxwhisker((
        x: 1,
        min: stormCSV.mse.at(0),
        max: stormCSV.mse.at(stormCSV.mse.len() - 1),
        q1: arrayPercentile(stormCSV.mse, 0.25),
        q2: arrayMedian(stormCSV.mse),
        q3: arrayPercentile(stormCSV.mse, 0.75),
      ))
      plot.add-boxwhisker((
        x: 2,
        min: convtasnetCSV.mse.at(0),
        max: convtasnetCSV.mse.at(convtasnetCSV.mse.len() - 1),
        q1: arrayPercentile(convtasnetCSV.mse, 0.25),
        q2: arrayMedian(convtasnetCSV.mse),
        q3: arrayPercentile(convtasnetCSV.mse, 0.75),
      ))
    },
  )
})
