#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ
#import "/thesis/utils/author.typ": *
#import "@preview/diagraph:0.3.6": *

= Methodology

== Dataset

Other machine learning fields mainly @CV and @LLM have long been trained on publically available diverse datasets @dengImageNetLargeScaleHierarchical2009 namely mC4, MassiveText or the Wikipedia dataset @naveedComprehensiveOverviewLarge2025.

As shown in @related_work previous work in the field of audio dereverberation has generally focused on speech signals. As this limitation is the same for many audio based machine learning problems (e.g. multi speaker seperation, noise cancellation and speech to text) many of the most used large audio datasets, like LibriSpeech or WSJ0, consist only of speech signals which are reduced in bandwidth as well as language diversity and recored in anechoic conditions @garofolojohns.CSRIWSJ0Complete2007 @panayotovLibrispeechASRCorpus2015 @richterEARSAnechoicFullband2024.

Datasets of diverse audio signals have emerged from audio classification problems. Early examples being private self collected datasets of indivdual researchers @woodardModelingClassificationNatural1992 @ellisDetectingAlarmSounds2001.
Over the recent years interest in audio classification has surged as can be seen in the amount of entries in the @DCASE challenge series that increased from 31 in 2013 to 428 in 2023 @mesarosDecadeDCASEAchievements2024. The @DCASE has also been a major influence in the increase of publically available datasets as prior to the @DCASE challenges only a limited amount were available most notably RWCP @smithPhysicalAudioSignal2010.

The current largest dataset of diverse audio signals is Google's fittingly named AudioSet containing over 5,800 hours of audio recordings with 527 classes
of annotated sounds @gemmekeAudioSetOntology2017. These recordings are 10 second clips drawn from YouTube videos. Building on top of the AudioSet classes the FSD50K dataset contains 100 hours of audio composed of 51,197 individual samples @fonsecaFSD50KOpenDataset2022 taken from the "freesound.org" audio sharing site. The FSD50K dataset is publically available while AudioSet released embedding features of the raw audio data necessitating a private download from YouTube. Both datasets are human-labeled while AudioSet specifies that sounds are human-verified and classes are suggested using YouTube metadata.

=== Data Collection<data_collection>

Our proposed approach requires a diverse dataset of dry audio data. In total 108,775 indivdual audio samples were collected resulting in the following dataset:

#figure(caption: [Dataset composition], table(
  columns: 3,
  align: (left, center, center),
  [*Dataset*], [*Number of Files*], [*Length of Files*],
  [_AudioSet_], [10790 (9.92 %)], [44h 52m 34s (13.8 %)],
  [_FSD50K_], [46753 (42.98 %)], [107h 34m 25s (33.1 %)],
  [_LibriSpeech_], [51232 (47.1 %)], [172h 17m 49s (53.1 %)],
  [_Total_], [108775], [324h 44m 49s],
))<dataset_comp>

Diverse audio data from the AudioSet and FSD50K datasets were downloaded in 44.1 kHz. Both datasets were used as to eliminate any bias occurring in one of the datasets (e.g. YouTube compression artifacts). The LibriSpeech dataset @panayotovLibrispeechASRCorpus2015 includes english utterances recored in anechoic conditions and sampled at 16 kHz. These were included in hopes of giving speech signals a greater weight as we felt clean speech was underrepresented in the other datasets.

A final dataset split of $70%$ training, $15%$ validation and $15%$ testing data was decided. Each sample was randomly assigned to one subset allowing for equal distribution of the entire dataset (cf. @dataset_comp) in each subset.

To assure reproducibility the file name of each sample is hashed using MD5 @rivestMD5MessagedigestAlgorithm1992:

$
  h & = op("MD5")(italic("name")) \
  r & = frac(op("int")(h_(0 dots 3))_"LE", 2^32 - 1)
$

The first 4 bytes are then used to assign the split:

$
  "split"(r) = cases(
    "train" & quad "if" r < S_"train",
    "val" & quad "if" S_"train" <= r < S_"train" + S_"val",
    "test" & quad "otherwise"
  )
$

With $S_"train"=0.7 "and" S_"val"=0.15$.

Another dataset of @RIR:pl was gathered, which was later in part used for reverberation purposes. The @RIR:pl were collected from the @AIR dataset @jeub09a as well as from the "Hybrid Reverb" plugin in Ableton Live 12 #footnote[https://www.ableton.com/en/packs/hybrid-reverb/]. The @AIR dataset contains 433 individual @RIR:pl with different acoustical properties, such as reverberation time and room volume.


// - RiR for TASNet training @jeub09a
// - what classes are covered
//   - look at PANNs paper for AudioSet Citation
// - own downloader, scraped from youtube in 44.1 kHz, talk about tech used and the theoretical quality possible
//   - IN THEORY ILLEGAL: DMCA 1201 / (Urheberrechtsgesetz) § 95a Schutz technischer Maßnahmen
// - talk about size and what we managed to download

=== Data Preprocessing

A supervised training approach (as explained in @supervised_learning) was chosen to train both our dereverberation (cf. @derev_process_pipeline) and perceptual loss model (cf. @percep_process_pipeline). Labeling was done automatically through synthetic reverberation of the dry audio samples included in the dataset described in @data_collection.

#figure(caption: [Dereverberation Preprocessing Pipeline], raw-render(
  ```dot
    digraph pipeline {
      rankdir=LR
      node [fontsize=10, style=filled, shape=box, rounded=true, width=1.8, height=0.4]
      edge [fontsize=8]

      dry   [fillcolor="white"]
      conv  [fillcolor="white"]
      rev   [fillcolor="white"]
      model [fillcolor="white"]

      {rank=same; conv; dry}

      dry  -> conv
      conv -> rev [constraint=false]
      dry  -> model [style=dashed, label="target signal", constraint=false]
      rev  -> model [label="input signal"]
    }
  ```,
  labels: (
    dry: [*Dry Audio*],
    conv: [*Reverberation*],
    rev: [*Reverberant Audio*],
    model: [*Model Training*],
  ),
))<derev_process_pipeline>


Additional labels used for the perceptual loss model (see @loss_network) were saved during parameter based reverberation (see @preprocessing_reverberation) and calculated from the dry-reverberant-sample pairs (see @preprocessing_peaq).

#figure(caption: [Perceptual Loss Preprocessing Pipeline], raw-render(
  ```dot
    digraph pipeline {
      rankdir=LR
      node [fontsize=10, style=filled, shape=box, rounded=true, width=2, height=0.4]
      edge [fontsize=8]

      dry   [fillcolor="white"]
      rev   [fillcolor="white"]
      model [fillcolor="white"]
      peaq  [fillcolor="white"]

      {rank=same; rev; dry; peaq}

      dry  -> model [label="signal a", constraint=false]
      dry -> rev
      rev  -> model [label="signal b"]
      rev -> model [style=dashed, label="wetness, size"]
      rev -> peaq
      dry -> peaq
      peaq -> model [style=dashed, label="odg"]
    }
  ```,
  labels: (
    dry: [*Dry Audio*],
    rev: [*Reverberant Audio*],
    model: [*Model Training*],
    peaq: [*PEAQ*],
  ),
))<percep_process_pipeline>

This synthetic labeling approach is similar in concept to self-supervised training where a supervisory signal is generated through augmentation. For instance in @CV:long tasks self-supervision is often used for autoencoder training or classification. Even in the domain of computational audio self-supervised approaches have shown great efficiency @baevskiWav2vec20Framework2020. However as our objective is neither autoassociative nor contrastive but a supervised regression from reverberant to dry audio it cannot be classified as such (see @self_supervised).

==== Reverberation<preprocessing_reverberation>

To provide the model with reverberant audio signals two kinds of preprocessing approaches were considered. The signals could either be reverberated  _"live"_, after loading a sample into memory during training, or _"offline"_ beforehand saving compute time but sacrificing disk space.

The three main ways of digital reverberation are @schlechtFeedbackDelayNetworks2018:
- convolutional
- delay networks
- computational acoustics

Reverberation through convolution via @RIR:pl is the most realistic way of generating synthetic reverb, as it mimics the scattering characteristics of a real-world room at the @RIR:pl recording position @farinaImpulseResponseMeasurements2007. Generally this comes at a higher computational cost and latency @siddiqOptimizationConvolutionReverberation2020 @misicAnalysisCPUGPU2016. Unfortunately convolution reverbs do not expose many parameters or controls, making labeling of samples which are fed to our loss network (see @loss_network) difficult.

Parameter based reverberation, like delay networks, is fast and requires little memory, but careful tuning is necessary to find configurations that sound realistic @schlechtFeedbackDelayNetworks2018 @siddiqOptimizationConvolutionReverberation2020. This gives us easy access to, e.g., size and wetness controls that we can use for labeling (see @loss_network).

In computational acoustics room simulations are used for reverberating audio @lemercierStoRMDiffusionbasedStochastic2023. Game engines such as Unity @mannallRoomAcoustiCOpensourceRoom2025 or libraries like pyroomacoustics @scheiblerPyroomacousticsPythonPackage2018 can be used to simulate rooms with different sizes, materials and microphone placements. This is done either by trying to solve the wave-equation by the discretization of the space, geometric solutions like the Image Source Method (ISM) @allenImageMethodEfficiently1979 or ray tracing @vorlanderAuralizationFundamentalsAcoustics2008. While this is attempting to recreate an acoustic space as close as possible, it is also the most computationally expensive and not possible to do live or offline for our amount of data. Unity's processing is also done in real time, which makes it not feasible, as the runtime would be about 324 hours (cf. @dataset_comp).

A first implementation was done using live processing in memory. All three approaches described above were implemented for an interchangable framework. Using this the original Conv-TasNet dataloader was adjusted to use the @RIR implementation.

After getting access to the RWTH Aachen CLAIX compute cluster, we pivoted to an offline dataset as compute time was of more importance than disk space. As mentioned parameter based reverberation was suited best for our own networks, due to the access to size and wetness controls.

Specifically, we first used Valhalla Supermassive in VST3 format, which was later abandoned, as it lacked Linux compatibility. We then chose a FreeVerb implementation @smithPhysicalAudioSignal2010 in the `pedalboard` Python package @sobotPedalboard2023.

// - two kinds of preprocessing either live during training in memory saving on disk space or before "offline" saving on compute during training but sacificing disk space
// - first implementation was in memory
//   - reverberation through room simulations, RIRs and Parameter reverb were implemented
//   - using this the original conv tasnet dataloader was adjusted to use the in memory RIR implementation
// - after getting access to online compute (with TB+ space) we ditched in memory approach as training compute time was of more importance
// - VST based parameter reverb is os dependent and make for a horrible workflow
// - live implementation as well as rir implementation for training of conv tasnet
// - reverb done with parameter reverb
//   - from pedalboard (FreeVerb implementation) @smithPhysicalAudioSignal2010

// - discuss ways considered to reverberate
//   - convolution with RIRs (room impulse responses)
//     - from open datasets like AIR @jeub09a
//     - most realistic but also most computationally expensive and not very flexible in terms of size and wetness control
//   - parameter based reverberation
//     - mostly used in music production as stylistic effect
//     - more flexible in terms of size and wetness control, less computationally expensive
//     - see fundamentals chapter
//   - room simulation in Unity or pyroomacoustics
//     - most realistic and flexible but also most computationally expensive and not possible to do offline for our amount of data
//     - Unity is done in realtime (add unity screenshots) -> not feasable for our amount of data (name total length of data in hours)
//     - pyroomacoustics is not realtime but also not possible to do offline or live for our amount of data

When processing the data from our dataset, the desicion was made to reverberate all files in their native sample rate and then later upsample them to the sample rate used for training. The three sub-datasets have the following sample rates: AudioSet at 44.1 kHz, LibriSpeech at 16 kHz, and Freesound at 44.1 kHz.

While 44.1 kHz is a fairly standard sample rate for consumer audio content @puAudioCompression2006, 16 kHz only allows for an upper frequency of $ f_"max" = f_s / 2 = (16 "kHz") / 2 = 8 "kHz" $ to be represented as shown by the Nyquist theorem @shannonCommunicationPresenceNoise1949. While this is technically enough to represent speech signals, which only need a bandwith of 300 Hz to 3400 Hz @itu-tG711PulseCode1988, we introduce some inconsistencies in the reverberation. The effects of these are discussed in @disc_upsampling.

// - sample rate: upscaling downscaling possible??
// - short usability study what sampling (higher limit) rates are possible in real world scenarios (DAC)

// - AudioSet is 44.1kHz: 10790 files
//   - theoretically not fully DRY audio
// - LibriMix/LibriSpeech is 16kHz: 51232 files
// - Freesound is 44.1kHz: 46753 files
//   - theoretically not fully DRY audio


// - saved precomputed values for :
//   - "size": np.interp(size, SIZE_RANGE, [0, 1]), #sym.arrow schon normiert
//   - "wetness": np.interp(wet, WET_RANGE, [0, 1]), #sym.arrow schon normiert



==== PEAQ<preprocessing_peaq>

As explained in @loss_network for every dry-reverberant-sample pair the @PEAQ scores @ODG and @DI (see @fun_peaq) were calculated. As the GStreamer implementation "GstPEAQ" was used @holtersGstPEAQOpenSource2015, GStreamer Python bindings were utilized to automate this process @GStreamerGstpython2026. This approach meant we needed both reference and test files written to disk making a live implementation not feasable. All samples were upsampled to 44.8 kHz for use with @PEAQ.

==== Non-Silent Parts

#let d_full = 819907050.9525146 / 1000 / 60 / 60
#let d_non_silent = 539188097 / 1000 / 60 / 60

Using the Python package Pydub @robertJiaaroPydub2026 non-silent ranges of all samples of the training subset (see @data_collection) were analyzed.
A sample is considered silent if its level is below $-40 "dBFS"$. A range of samples is considered silent once it is longer than 100 samples.

Using this formula $d_"full" approx #calc.round(d_full, digits: 2)$ hours of training data was examined. The duration of all non-silent ranges was $d_"non_silent" approx #calc.round(d_non_silent, digits: 2)$ hours leaving $d_"silent" approx #calc.round(d_full - d_non_silent, digits: 2)$ hours of silence or about

$ d_"silent"/d_"full" approx #calc.round(100 * (d_full - d_non_silent) / d_full, digits: 2) % $
. The problem is worsend by the fact that samples shorter than the segment length defined in @segment_length are zero padded to the desired length adding even more silent parts.

As we don't want our model to focus on generating silence a mask is generated for each sample specifying its silent ranges (cf. @silent_mask_signal). This mask is then used in the loss function to ignore the silent range (see @loss_function_silent_mask).

#figure(
  caption: [Signal with non-silent mask],
  image("/experiments/perceptual-quality/plots/mask_plot.svg"),
)<silent_mask_signal>


== Loss<loss_network>
#jojo

As described in @fun_loss_function a loss function is a qualitative function that is used to objectively measure model performance by calculating the deviation of the model's prediction to their ground truth counterpart. This deviation is mapped onto a real number that intuitively represents some error. To optimize model performance this error must be minimized.

As shown in @fun_loss_function different loss functions exist for different problem sets. Each research endeavor in machine learning must decide which loss function to use based on the nature of the problem, the data available and the type of machine learning algorithm to be solved @ciampiconiSurveyTaxonomyLoss2024.

In the time or waveform domain error-based regressive loss functions (e.g. @MSE, @SI-SNR and @PESQ) have identfied themselfs as well performing in the field of dereverberation (see @related_work and @fun_quality_metrics).

=== Analyzation of Applicable Loss Functions

The metrics described in @fun_quality_metrics can all be used a loss functions. The problem that all of them have in common it that non are specific to our task of dereverberation. @PESQ comes close beeing a perceptual scale- and shift-invariant metric but as it is made for the evaluation of speech signals, effectiveness in diverse audio signals is doubtful. #cite(<rixPerceptualEvaluationSpeech2001>, form: "prose", style: "chicago-author-date") write: "Certain other applications have not yet been fully characterised or may need parts of the model to be changed. These include: music quality [...]". An alternative lies in the @PEAQ:both model (cf. @fun_peaq).

Other state-of-the-art measures include Google's @ViSQOL @chinenViSQOLV3Open2020 as well as PEMO-Q @huberPEMOQANewMethod2006 (cf. @fun_quality_metrics). But comparison indicates that @PEAQ's performance is not only competitive but sometimes even superior to newer approaches @delgadoCanWeStill2020 meaning that @ViSQOL and PEMO-Q were not further considered.

As explained in @preprocessing_reverberation the final dataset was reverberated offline using an implementation of the FreeVerb reverberator allowing for export of size and wetness parameters on a per sample basis. The wetness and size parameters are objective measurements which we know to be true.
A fully reverberated signal is defined as:
$ ("wetness" = 1) and ("size"= 1) $
. A fully dereverberated signal is defined as:
$ ("wetness" = 1) or ("size"= 1) $
. This enables us to plot the different quality metrics against these objective measures and assess their applicability for the dereverberation task. Or in other words how well each metric estimates reverberation (and in turn dereverberation) of a signal.

#figure(
  caption: [Metrics usable as loss functions analyzed over 16421 datapoints from test dataset, data between the 15th and 85th percentile is shown in color],
  image("/experiments/perceptual-quality/plots/data_metrics_test_16421_15_85_percentile.svg"),
)











- as loss functions must be differentiable (see @fun_loss_function) it could be possible to use a neural network as a loss function. This would allow us to learn an algorithm that predicts a quality measure based on the objective values of size and wetness





We place the following requirements on the loss network:
- differentiable
- wideband (up to 44100 Hz)
- _good_ prediction of size and wetness parameters: use mse, corr metric here calculated by test script as indicator of good performance










- why nn as loss (better score for perceptual, combines perceptual and "real world" attribs)
- why mel scale not bark etc.
go through loss network and explain weights (quality, size, wetness, odg) etc. make links to how data was processed for this task

- cite similar papers in zotero loss subcollection (like LEAN, etc.) for fast audio classification
  - why our loss model was based on CNN14
  - runtime (inference) evaluation

- general comparison of different loss functions in audio ML (sisnr, pesq, mse, l1, our own)



#TODO[
  - Show \<15 and >85 data in e.g. grey
  - MSE: shit, HIGH DEVIATION FROM REGRESSION = bad
  - MAE: eh, ok for wetness, although extreme FEATHERING/spread out
  - corr: very good for wetness, weird range 0.8 is still technically correlating
  - PUT QUALTIY IN ANOTHER PLOT
    - make argument for why first quality score was using peaq (just because it should theoretically be the best)
    - also storm was using pesq: ALSO ADD PESQ TO PLOT FOR COMPARISON WITH PEAQ
    - then make argument why we removed PEAQ from quality score: in actuallity "bad" performance
    - SISNR is hard to normalize which is theoretically not needed for loss function, value range makes sense here (unlike corr or ODG norm)
  - maybe a plot for PESQ for only speech signals for reverberation
  - PEAQ is just not a good metric for reverberation
]

Key takeaways:
- wetness and size are objective measurements which we know to be true: $lim_("wet"arrow 1)$ and $lim_("size"arrow 1)$ means the signal is badly reverberated and $lim_("wet"arrow 0)$ and $lim_("size"arrow 0)$ means the signal is dereverberated
- correlation, mae and mse are bad loss functions as they do not accurately predict wetness or size values
- odg shows more "bad" (close to 0) values around higher wetness or size values, which is what we "need" from a loss function
- di does it similarily but we cannot normalize it that well
- si snr could also be used but experiments with tasNet showed even it inferior or close to just the standard mse
- train network on combination of odg, size and wetness resulting in quality score (lowest graph), which accurately predicts size and wetness


quality is here defined as:
$ Q = "ODG"_"norm" dot (1 - "wet"_"norm" dot 0.4) dot (1 - "size"_"norm" dot 0.3) $

- plot is little pointless here: akin to plotting wetness and size against theirselfs, BUT in the end this quality function will be estimated using Neural Network


LOSS Net is based on CNN14 as shown in PANNs paper. Originally for near real time audio tagging => made sense to use here.
