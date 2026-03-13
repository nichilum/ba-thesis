#import "/thesis/utils/todo.typ": TODO

= Evaluation

== Perceptual Quality Net<eval_percep_quality_net>

#figure(
  caption: [Quality score prediction analyzed over 16421 datapoints from test dataset (cf. @subset_comp), data between the 15th and 85th percentile is shown in color],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_quality.svg"),
)<plot_nn_qual_against_size_and_wet>

- ReLU: not entirely differentiable

== DISCUSS UPSAMPLING FOR TRAINING AND REVERBERATING AT LOWER (USING PLOTS)<disc_upsampling>

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model

== (An)echoic dataset
- AudioSet and FSD50K are not "dry" datasets. they contain samples that are recorded under echoic conditions.
- The model is not shown fully dereverberated sample pairs during training

== SI-SNR Calculations
- we did some fucky wucky here
- maybe thats why training Conv-TasNet with SI-SNR loss did not work as expected
