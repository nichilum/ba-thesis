#import "/thesis/utils/todo.typ": TODO

= Evaluation

== Perceptual Quality Net<eval_percep_quality_net>

=== Initial Implementation<eval_percep_qual_net_init>

- evaluated against cnn14 impl (best epoch from 10):

cn14:
Type,MSE,MAE,Correlation
quality,0.00586603581905365,0.03963662311434746,0.9115471243858337
odg,0.008650489151477814,0.052390482276678085,0.9287785887718201
size,0.0145123191177845,0.08344431966543198,0.9140910506248474
wetness,0.014580730348825455,0.08360379934310913,0.9138118028640747

initial:
Type,MSE,MAE,Correlation
quality,0.008666266687214375,0.04721766337752342,0.8671460151672363
odg,0.01355697587132454,0.061346303671598434,0.892558753490448
size,0.02266417257487774,0.1116771548986435,0.856648325920105
wetness,0.02280745655298233,0.11195141077041626,0.8552049398422241


=== CNN14<eval_percep_qual_net_cnn14>

#figure(
  caption: [Quality score prediction analyzed over 16421 datapoints from test dataset (cf. @subset_comp), data between the 15th and 85th percentile is shown in color],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile_quality.svg"),
)<plot_nn_qual_against_size_and_wet>

- @ReLU not entirely differentiable

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
