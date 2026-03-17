#import "/thesis/utils/todo.typ": TODO

= Evaluation

== Perceptual Quality Net<eval_percep_quality_net>

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

#figure(
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

@eval_init_vs_cnn14_compare shows improvements over all comparative metrics, Similar advancements have been made across all parameters (cf. @eval_init_vs_cnn14_compare_all).

#figure(
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

#figure(
  caption: [Quality score prediction analyzed over 16421 datapoints from test dataset (cf. @subset_comp), data between the 15th and 85th percentile is shown in color],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_quality.svg"),
)<plot_nn_qual_against_size_and_wet>

- ReLU: not entirely differentiable

#figure(
  caption: [Prediction quality of perceptual net from signal with increasing zero percentage],
  image("/experiments/perceptual-quality/plots/perceptual_net_zeros_preds.svg"),
)

== Objective Quality Net<eval_objective_quality_net>

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
