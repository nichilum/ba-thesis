#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/diagram.typ": diagram
#import "/thesis/utils/author.typ": *
= Evaluation

== Analyzation of Applicable Loss Functions

- discussion: is the size parameter a good metric to evaluate dereverberation. akin to making the room smaller, wetness "makes the oroginal signal louder"/more absorption in the room -> explain with mental model of room and mic

== Perceptual Quality Net<eval_percep_quality_net>
#jojo
Evaluation of the perceptual quality network is done both for the initial simple implementation as well as the CNN14 based one. The evaluation of the simple @CNN (see @eval_percep_qual_net_init) focuses on the comparison of the two approaches, examining the improvement the CNN14 based implementation made. @eval_percep_qual_net_cnn14 discusses general applicability of the perceptual quality network as loss function and its predictive performance compared to other metrics like the @SI-SNR.

=== Simple @CNN:short<eval_percep_qual_net_init>

This evaluation was carried out using a subset of the dataset discussed in @data_collection. Both the simple @CNN and the CNN14 based implementation were trained on the same subset. Training was done over 10 epochs saving only the best epoch of the training process.

The output of all prediction heads are compared to their ground truth counterpart using @MSE, @MAE and correlation as comparative metrics. @eval_init_vs_cnn14_compare shows absolute differences between ground truth and prediction as well as relative improvements between the simple @CNN and the CNN14 based implementation.

#let q_mse_init = 0.008666266687214375
#let q_mae_init = 0.04721766337752342
#let q_corr_init = 0.8671460151672363

#let q_mse_cnn14 = 0.00586603581905365
#let q_mae_cnn14 = 0.03963662311434746
#let q_corr_cnn14 = 0.9115471243858337


#let o_mse_init = 0.01355697587132454
#let o_mae_init = 0.061346303671598434
#let o_corr_init = 0.892558753490448

#let o_mse_cnn14 = 0.008650489151477814
#let o_mae_cnn14 = 0.052390482276678085
#let o_corr_cnn14 = 0.9287785887718201


#let s_mse_init = 0.02266417257487774
#let s_mae_init = 0.1116771548986435
#let s_corr_init = 0.856648325920105

#let s_mse_cnn14 = 0.0145123191177845
#let s_mae_cnn14 = 0.08344431966543198
#let s_corr_cnn14 = 0.9140910506248474


#let w_mse_init = 0.02280745655298233
#let w_mae_init = 0.11195141077041626
#let w_corr_init = 0.8552049398422241

#let w_mse_cnn14 = 0.014580730348825455
#let w_mae_cnn14 = 0.08360379934310913
#let w_corr_cnn14 = 0.9138118028640747

#diagram(
  caption: [Metric comparison of the quality score, taken from the best epoch of the first 10, between the simple @CNN and CNN14 implementation],

  table(
    columns: 4,
    align: (left, center, center, right),
    [Metric], [Simple @CNN], [CNN14], [Relative improvement],
    [@MSE],
    [#calc.round(q_mse_init, digits: 5)],
    [#calc.round(q_mse_cnn14, digits: 5)],
    [#calc.round((q_mse_init - q_mse_cnn14) / (q_mse_init) * 100, digits: 2) %],

    [@MAE],
    [#calc.round(q_mae_init, digits: 5)],
    [#calc.round(q_mae_cnn14, digits: 5)],
    [#calc.round((q_mae_init - q_mae_cnn14) / (q_mae_init) * 100, digits: 2) %],

    [Correlation],
    [#calc.round(q_corr_init, digits: 5)],
    [#calc.round(q_corr_cnn14, digits: 5)],
    [#calc.round((q_corr_cnn14 - q_corr_init) / (q_corr_init) * 100, digits: 2) %],
  ),
)<eval_init_vs_cnn14_compare>

@eval_init_vs_cnn14_compare shows improvements over all comparative metrics. Similar advancements have been made across all parameters (cf. @eval_init_vs_cnn14_compare_all).

#diagram(
  caption: [Relative improvement from the CNN14 implementation as compared to the simple @CNN implementation over all metrics and parameters, taken from the best epoch of the first 10],

  table(
    columns: 5,
    align: (left, center, center, center),
    [Metric], [quality score], [@ODG], [size], [wetness],
    [@MSE],
    [#calc.round((q_mse_init - q_mse_cnn14) / (q_mse_init) * 100, digits: 2) %],
    [#calc.round((o_mse_init - o_mse_cnn14) / (o_mse_init) * 100, digits: 2) %],
    [#calc.round((s_mse_init - s_mse_cnn14) / (s_mse_init) * 100, digits: 2) %],
    [#calc.round((w_mse_init - w_mse_cnn14) / (w_mse_init) * 100, digits: 2) %],

    [@MAE],
    [#calc.round((q_mae_init - q_mae_cnn14) / (q_mae_init) * 100, digits: 2) %],
    [#calc.round((o_mae_init - o_mae_cnn14) / (o_mae_init) * 100, digits: 2) %],
    [#calc.round((s_mae_init - s_mae_cnn14) / (s_mae_init) * 100, digits: 2) %],
    [#calc.round((w_mae_init - w_mae_cnn14) / (w_mae_init) * 100, digits: 2) %],

    [Correlation],
    [#calc.round((q_corr_cnn14 - q_corr_init) / (q_corr_init) * 100, digits: 2) %],
    [#calc.round((o_corr_cnn14 - o_corr_init) / (o_corr_init) * 100, digits: 2) %],
    [#calc.round((s_corr_cnn14 - s_corr_init) / (s_corr_init) * 100, digits: 2) %],
    [#calc.round((w_corr_cnn14 - w_corr_init) / (w_corr_init) * 100, digits: 2) %],
  ),
)<eval_init_vs_cnn14_compare_all>

These findings motivated us to proceed with the CNN14 based implementation as described in @impl_percep_qual_net_cnn14 for all further experiments.

=== CNN14<eval_percep_qual_net_cnn14>

#diagram(
  caption: [Quality score prediction analyzed over 16421 datapoints from test dataset (cf. @subset_comp), data between the 15th and 85th percentile is shown in color],
  short-caption: [Quality score prediction analyzed over test dataset],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_quality.svg"),
)<plot_nn_qual_against_size_and_wet>

- @ReLU not entirely differentiable

#diagram(
  caption: [Prediction quality of perceptual net from signal with increasing zero percentage],
  image("/experiments/perceptual-quality/plots/perceptual_net_zeros_preds.svg"),
)

- das der datensatz hall beinhaltet ist konzeptuell vlt schlimmer fur das quality net
- dass das quality net keine Referenz mehr braucht haben wir gar nicht ausgenutzt
  - im grunde hätten wir auf viel mehr daten von zb audioset die reverb beinhalten trainieren können
    - halt ohne make data ausguführen und alles voll zu müllen

== Objective Quality Net<eval_objective_quality_net>

- eval: es wäre interessant gewesen mal nur auf wetness zu trainieren

== Comparison of Conv-TasNet and StoRM for diverse signals
#leo
Both Conv-TasNet and StoRM were trained exclusively on speech recordings and have no exposure to music, environmental noise, or other non-speech content. To assess how each model generalises outside this group, both were applied to 2048 randomly sampled AudioSet clips spanning a wide range of acoustic scenes and event categories. @conv_tasnet_storm_comparison summarises these metrics, while the full distributions are shown in @boxplot_comparison.

#import "/thesis/content/results.typ": convtasnetCSV, d, stormCSV, v

#diagram(
  caption: [
    Comparison of different metrics for the evaluation of dereverberation performance of diverse audio samples, evaluated on 2048 random AudioSet samples. The runtime is measured on a single H100 GPU (CLAIX-2023-ML) @CLAIX2023RWTHHigh with 2048 Random AudioSet Samples.
  ],
  {
    // set text(size: 7pt)
    table(
      columns: 3,
      align: (left, center, center),
      [*Network*], [StoRM], [Conv-TasNet],
      [*MSE*], v(stormCSV.mse, digits: 5, std_digits: 3), v(convtasnetCSV.mse, digits: 5, std_digits: 3),
      [*SI-SNR (dB)*], v(stormCSV.si_snr), v(convtasnetCSV.si_snr),
      [*PESQ-WB*], v(stormCSV.pesq_wb), v(convtasnetCSV.pesq_wb),
      [*PESQ-NB*], v(stormCSV.pesq_nb), v(convtasnetCSV.pesq_nb),
      [*ODG*], v(stormCSV.odg), v(convtasnetCSV.odg),
      [*DI*], v(stormCSV.di), v(convtasnetCSV.di),
      [*Runtime (hh:mm:ss)*], [06:14:51], [00:04:15],
      [*Runtime per sample (s)*], [10.98], [0.12],
    )
  },
)<conv_tasnet_storm_comparison>

Across all metrics both models perform substantially below their in-domain speech statistics. PESQ-WB reaches only $1.35$ (StoRM) and $1.45$ (Conv-TasNet), far below StoRM's in-domain speech result of $2.83$ (@storm_paper_metrics). @ODG values of $-3.67$ and $-3.83$ place both models near the lower end of the five-step degradation scale, indicating consistently "annoying" to "very annoying" perceived quality. The boxplots in @boxplot_comparison confirm that these results are not driven by a few extreme samples: distributions are broad but consistens, with no single extreme point pulling results in one direction. Boxplots for @SI-SNR and @PESQ show some narrower interquartile ranges and shorter whiskers for the StoRM model.
A recurring informal observation from listening tests and viewing spectograms is that both models tend to lower the output level relative to the input, especially for non-speech signals. This unintended effect is visible in the spectrograms (@spectrogram_comparison, @spectrogram_comparison_storm) and likely contributes to the degraded metric values.

#diagram(
  caption: [Boxplot comparison of different metrics for the evaluation of dereverberation performance of diverse audio samples. (Outliers not shown)],
  short-caption: [Boxplot comparison for dereverberation performance of diverse audio samples],
  grid(
    columns: 2,
    column-gutter: 1cm,
    row-gutter: .5cm,
    align: right,
    d(stormCSV.mse, convtasnetCSV.mse, "MSE"), d(stormCSV.si_snr, convtasnetCSV.si_snr, "SI-SNR (dB)"),
    d(stormCSV.pesq_wb, convtasnetCSV.pesq_wb, "PESQ-WB"), d(stormCSV.pesq_nb, convtasnetCSV.pesq_nb, "PESQ-NB"),
    d(stormCSV.odg, convtasnetCSV.odg, "ODG"), d(stormCSV.di, convtasnetCSV.di, "DI"),
  ),
)<boxplot_comparison>

#TODO[Think about outliers in boxplots (only show some?)]

When comparing the two models against each other, the differences across most metrics are small relative to the standard deviation. Conv-TasNet achieves a marginally lower MSE ($0.028$ vs. $0.030$) and slightly higher PESQ-WB ($1.45$ vs. $1.35$), while StoRM scores better to a slight extent on ODG ($-3.67$ vs. $-3.83$) and DI ($-3.14$ vs. $-3.50$), suggesting it introduces fewer perceptual artifacts per sample on average. PESQ-NB is effectively equal ($1.82$ vs. $1.81$).

#TODO[talk about SI-SNR, only metric left undiscussed]

The most unambiguous differentiator is computational cost: at comparable out-of-domain performance, Conv-TasNet processes all 2048 samples in $4$ m $15$ s, while StoRM requires $6$ h $14$ m $51$ s --- approximately $88 times$ the inference time.
#TODO[time per sample from @conv_tasnet_storm_comparison, say how long each sample is]

== DISCUSS UPSAMPLING FOR TRAINING AND REVERBERATING AT LOWER (USING PLOTS)<disc_upsampling>

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model

== (An)echoic dataset
- AudioSet and FSD50K are not "dry" datasets. they contain samples that are recorded under echoic conditions.
- The model is not shown fully dereverberated sample pairs during training

== SI-SNR Calculations
- @conv_tasnet_loss_comparison and @conv_tasnet_storm_comparison
- did we fuck up here?
- maybe thats why training Conv-TasNet with SI-SNR loss did not work as expected
