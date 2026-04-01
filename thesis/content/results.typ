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

Switching to a standard @MSE loss resolved the issue: training converged stably to a validation loss of approximately 0.0009 after 125 epochs (cf. @conv_tasnet_loss_comparison). This is further discussed in @eval_si_snr_calculations.

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

Detailed results of the evaluation done on the @MSE\-trained model can be seen in @conv_tasnet_metrics. A positive @SI-SNR indicates dereverberation, while the @PESQ score still indicates a _"poor"_ performance (cf. @fun_pesq)

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
    align: (left, center, center, center, center),
    table.header([*Metric*], [*Mean*], [*Std*], [*Min*], [*Max*]),
    [SI-SNR], [10.036], [3.856], [-6.669], [19.226],
    // [SI-SDR], [10.003], [3.872], [-10.015], [19.220],
    [PESQ], [1.726], [0.304], [1.117], [3.015],
    [WV-MOS], [1.695], [0.441], [1.229], [3.350],
  ),
  caption: [Conv-TasNet dereverberation metrics on LibriSpeech `test-clean` (N = 2620)],
  short-caption: [Conv-TasNet dereverberation metrics on LibriSpeech],
)<conv_tasnet_metrics>

These limitations -- the 4 kHz bandwidth ceiling, speech-only training data, and the uncertainty of @SI-SNR as a training objective for diverse audio -- motivate the development of a dedicated dereverberation model trained on broadband diverse content and supported by a perceptual loss network.

== StoRM
#leo
Unlike Conv-TasNet, StoRM was not trained from scratch. We used the official pretrained dereverberation checkpoint provided by the authors, trained on the WSJ0 corpus reverberated with the REVERB challenge dataset @lemercierStoRMDiffusionbasedStochastic2023 @kinoshitaReverbChallengeCommon2013 @garofolojohns.CSRIWSJ0Complete2007. The training data consists of speech recordings sampled at 16 kHz, establishing an 8 kHz frequency ceiling. Architecturally, StoRM follows a generative stochastic regeneration approach: a discriminative denoiser first produces an initial estimate of the clean signal, which a score-based diffusion model then refines through a learned reverse process @lemercierStoRMDiffusionbasedStochastic2023. This contrasts with Conv-TasNet's discriminative masking, and the iterative inference required by the diffusion component has direct implications for computational cost.

On in-domain speech signals, StoRM achieves strong dereverberation quality. @storm_paper_metrics shows the evaluation from the original paper. These numbers serve as an upper bound for speech dereverberation quality achievable with this model.

#diagram(
  caption: [StoRM dereverberation metrics on the WSJ0+REVERB test set, reproduced from #cite(<lemercierStoRMDiffusionbasedStochastic2023>, form: "prose", style: "chicago-author-date").],
  short-caption: [StoRM dereverberation metrics on the WSJ0+REVERB test set],
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

All results presented in the following section are based on the dataset as described in @data_collection.
Results regarding prediction performance as well as use as a loss function are discussed. Results regarding the impact of the perceptual quality network on the dereverberation network are demonstrated in @results_derev_net.
Prediction performance was analyzed using the @MAE, @MSE and correlation (cf. @fun_quality_metrics). The average over the entire testing subset (cf. @subset_comp) is depicted in @results_percep_table.
As discussed in @eval_percep_qual_net_init all perceptual quality network experiments were conducted using the CNN14 based implementation. Therefore no results of the simple @CNN, as detailed in @impl_percep_qual_net_init, are shown going forward.

#let quality_net_metrics_table(csv_path) = {
  let data = csv(csv_path)

  table(
    columns: 4,
    align: (left, center, center, center),

    table.header([*Type*], [*MSE*], [*MAE*], [*Correlation*]),

    ..data
      .slice(1)
      .map(row => (
        row.at(0),
        [#calc.round(float(row.at(1)), digits: 6)],
        [#calc.round(float(row.at(2)), digits: 6)],
        [#calc.round(float(row.at(3)), digits: 6)],
      ))
      .flatten(),
  )
}

#diagram(
  quality_net_metrics_table("/experiments/perceptual-quality/plots/epoch_195-odg-perceptual_net_best.csv"),
  caption: [@MAE, @MSE and correlation of all perceptual quality metrics at epoch 195],
)<results_percep_table>

To visualize prediction performance @KDE plots as seen in @results_percep_pred_vs_truth were generated. Each prediction-head output (quality, wetness, size, @ODG) is plotted against its ground truth counterpart for each data-pair of the testing subset. Akin to the testing done in @analyze_loss_functions the quality score which was used as the loss function of the dereverberation network (see @impl_derev_net) was plotted against the real size, wetness and $"size" dot "wetness"$ values of the testing data (cf. @plot_nn_qual_against_size_and_wet). Further evaluation of these plots is found in @eval_percep_qual_net_cnn14.


#diagram(
  caption: [Quality score prediction of the perceptual quality network at epoch 195 analyzed as loss function over 16421 datapoints from the testing subset (cf. @subset_comp). Data between the 15th and 85th percentile is shown in color.],
  short-caption: [Perceptual quality network's quality score prediction analyzed as loss function over testing subset],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_quality.svg"),
)<plot_nn_qual_against_size_and_wet>

== Objective Quality Network<results_objective_quality_net>
#jojo

As described in @impl_objective_quality_network the objective quality network underwent a two stage refinement process. Retraining for 61 epochs using the updated quality score resulted in the metric averages as seen in @results_obj_score_table and the prediction versus ground truth plots as seen in @results_obj_score_pred_vs_truth.

#diagram(
  quality_net_metrics_table("/experiments/perceptual-quality/plots/epoch_61-quality-perceptual_net_best.csv"),
  caption: [@MAE, @MSE and correlation of all objective objective quality metrics using the updated quality score at epoch 61],
)<results_obj_score_table>

Retraining for 166 epochs using the update quality score as well as the update loss function calculation metrics averages as seen in @results_obj_score_loss_table and prediction versus ground truth plots as seen in @results_obj_score_loss_pred_vs_truth were achieved.

#diagram(
  quality_net_metrics_table("/experiments/perceptual-quality/plots/epoch_166-quality-perceptual_net_best.csv"),
  caption: [@MAE, @MSE and correlation of all objective objective quality metrics using the updated quality score and loss at epoch 166],
)<results_obj_score_loss_table>

It must be noted that with the updated loss function calculation which only uses the predicted quality score to adjust the models weights the results in @results_obj_score_loss_table and @results_obj_score_loss_pred_vs_truth for size, wetness and @ODG become meaningless.

Plotting the objective quality network as seen in @results_obj_score_loss_table against the corresponding size, wetness and $"size" dot "wetness"$ values resulted in the plot seen in @results_obj_score_loss_analyze. This visualization is discussed in @eval_objective_quality_net.

#diagram(
  caption: [Quality score prediction of the objective quality network using the updated quality score and loss at epoch 166 analyzed as loss function over 16421 datapoints from the testing subset (cf. @subset_comp). Data between the 15th and 85th percentile is shown in color.],
  short-caption: [Objective quality network's quality score prediction analyzed as loss function over testing subset],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_objective_score_lossy.svg"),
)<results_obj_score_loss_analyze>

Analyzing the inference speed of the objective quality network it was found that for a total length of 18.2 hours of audio data sampled at 44.1 kHz the total length of inference time amounted to about 7.74 seconds. As the objective quality network shares the same architecture with the perceptual quality network this result can be assumed similar for all of the above tested configurations.



// TOTAL NUMBER OF SAMPLES: 2896664400
// Total length of audio coded in 44.1 kHz is 18.2455555556 hours
// TOTAL LENGTH OF INFERENCE TIME: 7.738234307016683

== Dereverberation Network<results_derev_net>
#leo

As described in @impl_derev_net the dereverberation network was trained using multiple loss functions. Namely @SI-SNR as well as the perceptual quality network (@meth_percep_quality_net) and objective quality network (@meth_obj_quality_net) were tested.

First a baseline using the @SI-SNR was established. Training was done over 62 epochs with early stopping at epoch 55 resulting in a validation loss of 16.6097 dB (cf. @derev_tcn_v4_loss_SISNR).
@tab_derev_sisnr_results is showing performance metrics for model outputs and is further discussed as part of the evaluation in @eval_derev_net.

#figure(
  caption: [Evaluation metrics for the SI-SNR baseline model (epoch 55, validation loss =-16.6097 dB). PEAQ metrics computed on n = 1758 samples.],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    table.header([*Metric*], [*Baseline*], [*Enhanced*], [*Δ*]),
    [SI-SNR (dB)], [9.98 ± 8.67], [18.78 ± 7.01], [+8.80 ± 3.02],
    [PESQ], [1.88 ± 0.84], [2.94 ± 0.78], [+1.06 ± 0.38],
    [PEAQ ODG], [-3.57 ± 0.75], [-3.65 ± 0.34], [-0.11 ± 0.55],
    [PEAQ DI], [-2.92 ± 1.41], [-2.85 ± 0.92], [+0.01 ± 1.00],
    [MSE], [0.0010 ± 0.0031], [0.0216 ± 0.0233], [-0.0206 ± 0.0217],
    // [Inference (s)],  table.cell(colspan: 3)[0.0981 ± 0.0574 \ #text(size: 0.85em, fill: gray)[min 0.0137 · max 0.5153]],
    // [RTF],            table.cell(colspan: 3)[0.0098 ± 0.0041]
  ),
)<tab_derev_sisnr_results>


#grid(
  columns: 2,
  column-gutter: 1cm,
  [#diagram(
    caption: [
      Loss curve of the dereverberation network trained with @SI-SNR loss. Smoothed using an exponential moving average with $alpha=0.05$.
    ],
    short-caption: [Loss curve of the dereverberation network trained with @SI-SNR:short loss],
    image(
      "../figures/derevnet-derev_tcn_v4_loss_SISNR.svg",
    ),
  )<derev_tcn_v4_loss_SISNR>],
  [#diagram(
    caption: [
      Loss curve of the dereverberation network trained with the objective perceptual network. Smoothed using an exponential moving average with $alpha=0.05$.
    ],
    short-caption: [Loss curve of the dereverberation network trained with the objective perceptual network],
    image(
      "../figures/derevnet-derev_tcn_v4_loss_percep.svg",
    ),
  )<derev_tcn_v4_loss_percep>],
)

Results of the model output can be seen in @derev_tcn_v4_sisnr and @derev_tcn_v4_sisnr_updated, where @derev_tcn_v4_sisnr is a earlier checkpoint at 16 epochs and a validation loss of 14.3848 dB.

#diagram(
  caption: [
    Spectrogram comparison of input (left) and output (middle) of the dereverberation network trained with @SI-SNR loss at epoch 16 (validation loss = -14.3848 dB). The rightmost plot shows the original clean signal for reference.
  ],
  short-caption: [Spectrogram comparison of input and output of the dereverberation network trained with @SI-SNR:short loss at epoch 16],
  image(
    "../../experiments/perceptual-quality/test_output/derev_tcn_v4_SISNR-epoch=16-val_loss=-14.3848/spectrograms/353-128309-0032.png",
  ),
)<derev_tcn_v4_sisnr>
#diagram(
  caption: [
    Spectrogram comparison of input (left) and output (middle) of the dereverberation network trained with @SI-SNR loss at epoch 55 (validation loss = -16.6097 dB). The rightmost plot shows the original clean signal for reference.
  ],
  short-caption: [Spectrogram comparison of input and output of the dereverberation network trained with @SI-SNR:short loss at epoch 55],
  image(
    "../../experiments/perceptual-quality/test_output/derev_tcn_v4_SISNR-epoch=55-val_loss=-16.6097/spectrograms/353-128309-0032.png",
  ),
)<derev_tcn_v4_sisnr_updated>

Both the perceptual quality network and objective quality network were not successful in their application. Training and evaluation loss did both approach 0 (see @derev_tcn_v4_loss_percep for objective quality network), but showed no effective dereverberation. In fact both loss metrics failed to reconstruct the original signal properly, resulting in audible artifacts and visible coloration of the source material (cf. @derev_tcn_v4_percep).

#diagram(
  caption: [
    Spectrogram comparison of input (left) and output (middle) of the dereverberation network trained with the objective perceptual network at epoch 70 (validation loss = 0.0154). The rightmost plot shows the original clean signal for reference.
  ],
  short-caption: [Spectrogram comparison of input and output of the dereverberation network trained with the objective perceptual network at epoch 70],
  image(
    "../../experiments/perceptual-quality/test_output/derev_tcn_v4_percep-epoch=70-val_loss=0.0154/spectrograms/353-128309-0032.png",
  ),
)<derev_tcn_v4_percep>
