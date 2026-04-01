#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/diagram.typ": diagram
#import "/thesis/utils/author.typ": *
= Evaluation

This chapter discusses the results shown @results in regards to the problem statement found in @intro_problem.  Shortcomings of our approach introduced in @methodology are debated and further research questions are asked.

== Analyzation of Applicable Loss Functions<eval_analyze_loss_functions>

The analyzation of applicable loss functions carried out in @analyze_loss_functions bases itself on the assumption that the size and wetness parameters which were used to parametrize the reverb are good indicators of dereverberation performance. It is reasonable to assume that wetness is a good indicator as it is used as a factor to multiple the wet signal in the freeverb implementation. Therefore it has no impact on the reverb tail itself, only making it quieter which is what we expect of our model. The size parameter behaves differently. It indirectly adjusts the intensity of each delayed signal of the comb filters. It is unclear how well a reduction in room size corresponds with reducing reverberation.

@plot_metrics_against_size_and_wet shows every metric performs worse against the size parameter. This might not be the metrics fault, consequently all conclusions based on the size plots alone require further investigation.

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

The @MSE, @MAE and correlation averages shown in @results_percep_table indicate very good prediction performance of every prediction head on unseen data. Compared with Quality-Net (see @related_quality_net) we achieved an @ODG @MSE value of 0.006775, where the @ODG was normalized, while Quality-Net showed a @PESQ @MSE of 0.1266, with @PESQ values ranging from 0 to 4. Dividing Quality-Net's @MSE by 4 results in a normalized @MSE of $0.03165$ which is still notably worse than ours.

@results_percep_pred_vs_truth shows that the quality score as proposed in @meth_percep_quality_net is very dependent on the @ODG. This turned out to be a problem as @ODG values were unexpectedly overrepresented in the 0 to 0.2 range, meaning that most signals were classified as imperceptibly imparied. In theory this shouldn't impact the use of the network as a loss function but some signals were graded significanty worse (in the 0.8 to 1 range) making gradient descent based on the @ODG unpredictable. Wetness and size predictions behaved as expected with some skewing in the upper range (0.9 to 1). This problem can be observed in both plots although stronger impact is noticed in the size predictions. This observation lead to the stronger weighting of the wetness prediction in @meth_obj_quality_net.

@plot_nn_qual_against_size_and_wet plots the quality score against size, wetness and $"size" dot "wetness"$. This visualization has the same shortcomings as the ones in @analyze_loss_functions as explained in @eval_analyze_loss_functions. It is shown that compared with the @ODG the perceptual quality network has improved reverberation prediction performance. Unfortunately it cannot compete with the @SI-SNR (cf. @plot_metrics_against_size_and_wet).

@meth_silent_parts noted that about one third of our dataset was silence. It was therefore imporant to investigate whether model performance suffers from prolonged silence in an audio signal. To evaluate this a single audio file was loaded and quality, size and wetness predictions were made. Iteratively parts of the audio signal were replaced with zeros. @quality_net_perf_silence shows the predicted values with increasing zero percentage. Remarkably the model's prediction kept steady up to about $90 %$ zero percentage. It was therefore concluded that using the perceptual quality network as a loss function on signals with silent parts should not pose an issue.

#diagram(
  caption: [Prediction quality of perceptual net from signal with increasing zero percentage],
  image("/experiments/perceptual-quality/plots/perceptual_net_zeros_preds.svg"),
)<quality_net_perf_silence>

Multiple shortcomings of the general approach to the training and using as a loss function of the perceptual quality network were identified. As described in @data_collection our dataset is in parts composed of the AudioSet and FSD50K. The samples found in these datasets are not necessarily recorded in anechoic conditions, meaning that it is possible that an already reverberated file was reverberated with a wetness of 0 and therefore mislabled. As the FSD50K dataset is taken from a sample sharing site it is less likely this problem has occurred there. A related issue concerns the training concept, as we used the same dataset for both the training of the perceptual quality network and the dereverberation network. While both used the same subsets (cf. @subset_comp) it is possible that the perceptual quality network wrongly overfitted on some samples. Further research should also investigate the use of activation function @ReLU as differentiability is not guaranteed (cf. @meth_percep_quality_net).

== Objective Quality Net<eval_objective_quality_net>

The @MSE, @MAE and correlation averages of the quality score shown in @results_obj_score_table demonstrate worse performance than the ones of the perceptual quality net (cf. @results_percep_table). This is to be expected as now all values of the quality score are uniformly distributed in the range of 0 to 1.

As we did not utilize the singular @ODG, size and wetness predictions it was decided that the extra computational resources used to propagate errors stemming from these prediction heads can be saved. Using the updated loss function (cf. @impl_objective_quality_network) the metric averages of the quality score did not improve notably (cf. @results_obj_score_loss_table).

@results_obj_score_loss_analyze shows that the quality score stemming from the objective quality network using the updated loss function is competitive with the @SI-SNR metric for indicating dereverberation performance (cf. @analyze_loss_functions). Predictions against the size parameter have improved substantially, exhibiting markedly reduced error. While predictions against the wetness parameter remain more dispersed, the nonlinear curvature previously observed in the @SI-SNR visualization (cf. @plot_metrics_against_size_and_wet) is no longer present, indicating that optimization with respect to this metric should yield more stable and predictable convergence behavior.

Inference speed at 7.74 seconds for 18.2 hourse of audio data sampled at 44.1 kHz means that using this network as a loss function should not pose any significant drawbacks in regards to training speed.

Non of the shortcomings as identified in @eval_percep_qual_net_cnn14 were addressed or mitigated in any meaningful capacity.

== SI-SNR Calculations<eval_si_snr_calculations>
// - @conv_tasnet_loss_comparison and @conv_tasnet_storm_comparison
// - did we fuck up here?
// - maybe thats why training Conv-TasNet with SI-SNR loss did not work as expected

// - explain that the MSE checkpoint does return comparable SISNR values from the librispeech dataset as the original ConvTasNet paper reports
// - 15.3 dB in @luoConvTasNetSurpassingIdeal2019 vs 10.0 dB in @conv_tasnet_metrics

The @SI-SNR values reported in @conv_tasnet_loss_comparison and @conv_tasnet_storm_comparison warrant closer scrutiny. Specifically, when our Conv-TasNet re-implementation was trained using @SI-SNR loss, convergence did not proceed as expected, with the loss failing to decrease in a manner consistent with the improvements observed under @MSE training. This raises the question of whether the @SI-SNR calculations used for that training were correctly implemented and whether this accounts for the unexpected training behaviour.

To validate the implementation, the @MSE\-trained Conv-TasNet checkpoint was evaluated on the LibriSpeech test set and yielded an @SI-SNR of 10.0 dB. This falls short of the 15.3 dB reported in the original Conv-TasNet paper @luoConvTasNetSurpassingIdeal2019, a gap that is too large to attribute solely to differences in training data or hyperparameters. While the model was not trained under identical conditions to the original work, the magnitude of the discrepancy suggests that a systematic issue may be present in the @SI-SNR calculation. It should be noted that while the @MSE checkpoint does fall short in @SI-SNR compared to the original paper, it still demonstrates a significant improvement over the reverberant input, indicating that the model is learning to dereverberate to some extent.

== Comparison of Conv-TasNet and StoRM for diverse signals
#leo
Both Conv-TasNet and StoRM were trained exclusively on speech recordings and had no exposure to music, environmental noise, or other non-speech content. To assess how each model generalises outside this group, both were applied to 2048 randomly sampled AudioSet clips spanning a wide range of acoustic scenes and event categories. @conv_tasnet_storm_comparison summarises these metrics, while the full distributions are shown in @boxplot_comparison.

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

Across all metrics both models perform substantially below their in-domain speech statistics. PESQ-WB reaches only $1.35$ (StoRM) and $1.45$ (Conv-TasNet), far below StoRM's in-domain speech result of $2.83$ (@storm_paper_metrics). @ODG values of $-3.67$ and $-3.83$ place both models near the lower end of the five-step degradation scale, indicating consistently "annoying" to "very annoying" perceived quality. The boxplots in @boxplot_comparison confirm that these results are not driven by a few extreme samples: distributions are broad but consistent, with no single extreme point pulling results in one direction. Boxplots for @SI-SNR and @PESQ show some narrower interquartile ranges and shorter whiskers for the StoRM model, meaning Conv-TasNet's performance is more inconsistent.
A recurring informal observation from listening tests and viewing spectograms is that both models tend to lower the output level relative to the input, especially for non-speech signals. This unintended effect is visible in the spectrograms (@spectrogram_comparison and @spectrogram_comparison_storm) and likely contributes to the degraded metric values.

It should be noted that the @SI-SNR values obtained from our Conv-TasNet implementation trained with the MSE loss function may deviate from those reported by the original implementation (see @eval_si_snr_calculations), although a comparable disparity between in-domain and out-of-domain performance can be anticipated when using the original implementation under analogous conditions.

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

When comparing the two models against each other, the differences across most metrics are small relative to the standard deviation. Conv-TasNet achieves a marginally lower MSE ($0.028$ vs. $0.030$) and slightly higher PESQ-WB ($1.45$ vs. $1.35$), while StoRM scores better to a slight extent on ODG ($-3.67$ vs. $-3.83$) and DI ($-3.14$ vs. $-3.50$), suggesting it introduces fewer perceptual artifacts per sample on average. PESQ-NB is effectively equal ($1.82$ vs. $1.81$). @SI-SNR values are also close ($-31.39$ vs. $-31.26$) but with a notably higher standard deviation for both models, also indicating that performance is more inconsistent across samples.


The most unambiguous differentiator is computational cost: at comparable out-of-domain performance, Conv-TasNet processes all 2048 samples in $4$ m $15$ s ($0.12$ s per sample), while StoRM requires $6$ h $14$ m $51$ s ($10.98$ s per sample) --- approximately $91 times$ the inference time. The samples were each 10 s long, meaning that Conv-TasNet processes audio roughly $83times$ faster than real time, while StoRM operates at approximately $0.91times$ real time, just below the threshold for real-time applicability, and notably on comparatively powerful hardware.

Overall, the differences between the two models are minimal and likely not perceptually significant, with both struggling to generalise to non-speech content.

== DISCUSS UPSAMPLING FOR TRAINING AND REVERBERATING AT LOWER (USING PLOTS)<disc_upsampling>

#TODO[]

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model

== Dereverberation Network<eval_derev_net>

As mentioned in @impl_derev_net a first implementation of the dereverberation network followed a purely generative approach, without masking operations. To validate the capacity of the model a subset of the dataset was used to try and overfit this first implementation. This was successfully done as seen in @derevnet_derev_1_overfit_version_22_loss, which shows a steady decrease in training and validation loss. The training loss reaches a value of $0.00005$ and the validation loss of $0.0005$, which is a strong indication that capacity is sufficient to learn the dereverberation of the training data. However, we could not replicate this behaviour on the full dataset, meaning the model could not generalize well, which is why we switched to the masking approach described in @impl_derev_net.

#diagram(
  caption: [
    Loss curves of the first dereverberation network implementation, trained on a subset of the dataset using the perceptual quality network as a loss function.
  ],
  image(
    "../figures/derevnet-derev_1_overfit_version_22_loss.svg",
  ),
)<derevnet_derev_1_overfit_version_22_loss>

Evaluation of the dereverberation network using the objective quality network was carried out sparsely as the model's output was not producing meaningful results. This can be seen in @derev_tcn_v4_percep, where the spectrum in the center shows the output, which barely resembles the clean reference on the right. The signal is distributed across the full model bandwidth (22.05 kHz) adding high frequency noise, while the reverberation tail remains entirely unattenuated. The only discernible structure preserved from the clean signal is the gross syllabic rhythm; however, transient onsets are severely smeared across this otherwise incoherent spectrum. The precise cause of this phenomenon remains unclear. However, it can be hypothesized that, as discussed in @eval_percep_qual_net_cnn14, the shared training dataset between both models may be a contributing factor. Theoretically, a network exhibiting the performance characteristics shown in @results_objective_quality_net should be sufficient, as comparable models have been successfully employed in prior work @fuMetricGANImprovedVersion2021.

// - gating effect
// - adds highs in some examples
//   - makes sense because of "learning" at 44.1 kHz (@preprocessing_reverberation)
//   - "upsampling" effect almost pleasant in some cases, but also adds unwanted artifacts in others
//   - discussion: maybe reverberating at 44.1 kHz would have made sense (@disc_upsampling)
// - if the input signal is not so reverberant, the output only gets dereverberated very little
// - quality of dereverberation is highly dependent on the quality of the input signal


The dereverberation network using @SI-SNR was evaluated both using spectograms (@derev_tcn_v4_sisnr and @derev_tcn_v4_sisnr_updated) on a selected slice of the test set and on selected metrics (@tab_derev_sisnr_results), allowing comparison both with StoRM and Conv-TasNet.

In @derev_tcn_v4_sisnr_updated one can see a dereverberated speech signal in the center. Individual transients are clearly visible and the rhythmic structure of the syllables is preserved. Reverberation tails are reduced in both magnitude and duration, though not fully eliminated. The spectral envelope remains close to that of the clean reference, indicating that the model does not introduce significant spectral distortion. An audible gating or pumping effect is noticeable, likely caused by the model suppressing regions between transients.

A further observation is that the model introduces additional high-frequency energy in some examples. This is consistent with the network having been trained on signals upsampled to 44.1 kHz, as described in @preprocessing_reverberation. The model may have learned high-frequency patterns from the upsampling process rather than from genuine acoustic content. The resulting effect is occasionally perceived as a subtle enhancement, though it also introduces unwanted artefacts in other cases. This motivates the discussion in @disc_upsampling of whether reverberation should instead have been applied at the native 44.1 kHz prior to training.

One can see the effect of the additional #{ 55 - 16 } epochs of training when comparing @derev_tcn_v4_sisnr and @derev_tcn_v4_sisnr_updated. While the reverberation tail is only slightly further reduced in magnitude, the high-frequency noise is substantially reduced, but still present. This is reflected in the @SI-SNR improvement, which increases from $11.8$ dB to $16.2$ dB for the example shown.

The degree of dereverberation is closely tied to the characteristics of the input signal. Samples with little reverberation show only marginal improvement after processing, suggesting that the model scales its output to the perceived degree of reverberation in the input rather than applying a fixed transformation. More broadly, the perceptual quality of the enhanced signal is strongly dependent on the quality of the input. Clean or mildly reverberant signals tend to be processed faithfully, while heavily degraded inputs yield less consistent results.

@tab_derev_sisnr_results shows a substabtial @SI-SNR improvement of $+8.80$ dB on average, which is slightly worse than the validation loss of $16.61$ dB, but still confirms that the dereverberation network is able to generalise to unseen data. @PESQ improves from $1.88$ to $2.94$, an increase of $+1.06$, indicating a meaningful gain in predicted speech quality. Notably, the standard deviation of both @SI-SNR and @PESQ is reduced in the enhanced signal relative to the baseline, suggesting that the model produces more consistent outputs across varying degrees of input reverberation.

The @PEAQ metrics tell a different story. The @ODG decreases marginally by $-0.11$, and the @DI shows a negligible change of $+0.01$, with both deltas exhibiting standard deviations larger than the mean shift itself. This is contrary to the improvements observed in @SI-SNR and @PESQ, again raising the question of whether @PEAQ is a suitable metric for evaluating dereverberation performance, as discussed in @analyze_loss_functions. The @MSE increase from $0.0010$ to $0.0216$ is a direct consequence of the @SI-SNR training objective, which optimises signal-to-noise ratio rather than sample-wise reconstruction fidelity, and is therefore expected.

Informal listening tests conducted on a selected set of samples yield the unexpected finding that music signals are well-suited to the proposed dereverberation approach. While the structural integrity of the recordings is preserved, reverberation-induced transient extensions and elongated decay times of percussive elements are effectively attenuated. Notably, this improvement remains perceptually noticeable even in fully mixed recordings, where the percussive components are masked by concurrent instrumental layers.

When compared against StoRM and Conv-TasNet, the dereverberation network achieves markedly superior performance on diverse audio signals, a result that is consistent with expectations given the domain-specific training. However, it remains an open question whether Conv-TasNet or StoRM would yield comparable performance if trained exclusively on the same dataset.

#diagram(
  caption: [
    Comparison of dereverberation performance of our model against StoRM and Conv-TasNet across in-domain speech and out-of-domain diverse audio samples.
  ],
  short-caption: [Comparison of dereverberation performance of our model against StoRM and Conv-TasNet],
  table(
    // columns: (1fr, 1fr, 1fr),
    columns: 3,
    align: (left, right, right),
    table.header([*Model*], [*Speech*], [*Diverse*]),
    [Conv-TasNet],
    [15.3 $plus.minus$ ? dB #footnote[@SI-SNR taken from #cite(<luoConvTasNetSurpassingIdeal2019>, form: "prose", style: "chicago-author-date"), standard deviation not reported]],
    [-31.26 $plus.minus$ 18.6 dB],

    [StoRM], [6.5 $plus.minus$ 4.0 dB], [-31.39 $plus.minus$ 11.71 dB],
    [Ours], [-], [8.80 $plus.minus$ 3.02 dB],
  ),
)


Evaluation of the dereverberation network lead to the identification of multiple shortcomings. As discussed in @meth_percep_quality_net, the motivation for developing the perceptual and objective quality network extended beyond identifying an optimal indicator of dereverberation performance; it also eliminated the requirement for a clean reference signal during training. This improvement could have allowed us to train fully unsupervised on a larger subset of the AudioSet dataset, as it would have included many reverberant signals. This approach was not explored within the scope of this work.

The topic of reverberation being present in our dataset was touched upon in @eval_percep_qual_net_cnn14. It is discussed that the AudioSet and FSD50K datasets were not necessarily recorded in anechoic conditions. This can lead to mislabeling of data for use in training of the quality networks. Another problem arising is that the dereverberation network is shown echoic samples as ground truth therefore not learning to eliminate reverberation but only reducing it. In the worst case, this occurs in $46.9 %$ of instances (cf. @dataset_comp). It can be hypothesized that a considerable proportion of AudioSet samples originate from voiceover recordings, which are typically captured under near-anechoic conditions. Furthermore, FreeSound, as a sample-sharing platform, likely contains a high prevalence of acoustically dry recordings.

Further investigation is warranted to systematically compare datasets comprising exclusively anechoic recordings against those biased toward dry samples yet containing a non-negligible proportion of reverberant signals.

