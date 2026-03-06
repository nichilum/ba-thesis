#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ
#import "/thesis/utils/author.typ": *
#import "@preview/diagraph:0.3.6": *

= Methodology

== Dataset

Other machine learning fields mainly computer vision (CV) and large language models (LLMs) have long been trained on publically available diverse datasets @dengImageNetLargeScaleHierarchical2009 namely mC4, MassiveText or the Wikipedia dataset @naveedComprehensiveOverviewLarge2025.

As shown in @related_work previous work in the field of audio dereverberation has generally focused on speech signals. As this limitation is the same for many audio based machine learning problems (e.g. multi speaker seperation, noise cancellation and speech to text) many of the most used large audio datasets, like LibriSpeech or WSJ0, consist only of speech signals which are reduced in bandwidth as well as language diversity and recored in anechoic conditions @garofolojohns.CSRIWSJ0Complete2007 @panayotovLibrispeechASRCorpus2015 @richterEARSAnechoicFullband2024.

Datasets of diverse audio signals have emerged from audio classification problems. Early examples being private self collected datasets of indivdual researchers @woodardModelingClassificationNatural1992 @ellisDetectingAlarmSounds2001.
Over the recent years interest in audio classification has surged as can be seen in the amount of entries in the "Detection and Classification of Acoustic Scenes and Events" (DCASE) challenge series that increased from 31 in 2013 to 428 in 2023 @mesarosDecadeDCASEAchievements2024. The DCASE has also been a major influence in the increase of publically available datasets as prior to the DCASE challenges only a limited amount were available most notably RWCP @smithPhysicalAudioSignal2010.

The current largest dataset of diverse audio signals is Google's fittingly named AudioSet containing over 5,800 hours of audio recordings with 527 classes
of annotated sounds @gemmekeAudioSetOntology2017. These recordings are 10 second clips drawn from YouTube videos. Building on top of the AudioSet classes the FSD50K dataset contains 100 hours of audio composed of 51,197 individual samples @fonsecaFSD50KOpenDataset2022 taken from the "freesound.org" audio sharing site. The FSD50K dataset is publically available while AudioSet released embedding features of the raw audio data necessitating a private download from YouTube. Both datasets are human-labeled while AudioSet specifies that sounds are human-verified and classes are suggested using YouTube metadata.

=== Data Collection<data_collection>

Our proposed approach requires a diverse dataset of dry audio data. In total 108,775 indivdual audio samples were collected resulting in the following dataset:

#figure(caption: [Dataset split], table(
  columns: 3,
  align: (left, center, center),
  [*Dataset*], [*Number of Files*], [*Length of Files*],
  [_AudioSet_], [10790 (9.92 %)], [44h 52m 34s (13.8 %)],
  [_FSD50K_], [46753 (42.98 %)], [107h 34m 25s (33.1 %)],
  [_LibriSpeech_], [51232 (47.1 %)], [172h 17m 49s (53.1 %)],
  [_Total_], [108775], [324h 44m 49s],
))<dataset_split>

Diverse audio data from the AudioSet and FSD50K datasets were downloaded in 44.1 kHz. Both datasets were used as to eliminate any bias occurring in one of the datasets (e.g. YouTube compression artifacts). The LibriSpeech dataset @panayotovLibrispeechASRCorpus2015 includes english utterances recored in anechoic conditions and sampled at 16 kHz. These were included in hopes of giving speech signals a greater weight as we felt clean speech was underrepresented in the other datasets.

Another dataset of room impulse responses (RIR) was gathered, which was later in part used for reverberation purposes. The RIRs were collected from the Aachen Impulse Response (AIR) dataset @jeub09a as well as from the "Hybrid Reverb" plugin in Ableton Live 12 #footnote[https://www.ableton.com/en/packs/hybrid-reverb/]. The AIR dataset contains 433 individual RIRs with different acoustical properties, such as reverberation time and room volume.


// - RiR for TASNet training @jeub09a
// - what classes are covered
//   - look at PANNs paper for AudioSet Citation
// - own downloader, scraped from youtube in 44.1 kHz, talk about tech used and the theoretical quality possible
//   - IN THEORY ILLEGAL: DMCA 1201 / (Urheberrechtsgesetz) § 95a Schutz technischer Maßnahmen
// - talk about size and what we managed to download

=== Data Preprocessing

A supervised training approach (as explained in @supervised_learning) was chosen to train our model. Labeling was done automatically through synthetic reverberation of the dry audio samples included in the dataset described in @data_collection.

#figure(caption: [Augmentation pipeline], raw-render(
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
))

This synthetic labeling approach is similar in concept to self-supervised training where a supervisory signal is generated through augmentation. For instance in computer vision tasks self-supervision is often used for autoencoder training or classification. Even in the domain of computational audio self-supervised approaches have shown great efficiency @baevskiWav2vec20Framework2020. However as our objective is neither autoassociative nor contrastive but a supervised regression from reverberant to dry audio it cannot be classified as such (see @self_supervised).

==== Reverberation

To provide the model with reverberant audio signals two kinds of preprocessing approaches were considered. The signals could either be reverberated  _"live"_, after loading a sample into memory during training, or _"offline"_ beforehand saving compute time but sacrificing disk space.

The three main ways of digital reverberation are @schlechtFeedbackDelayNetworks2018:
- convolutional
- delay networks
- computational acoustics

Reverberation through convolution via @RIR:pl is the most realistic way of generating synthetic reverb, as it mimics the scattering characteristics of a real-world room at the @RIR:pl recording position @farinaImpulseResponseMeasurements2007. Generally this comes at a higher computational cost and latency @siddiqOptimizationConvolutionReverberation2020 @misicAnalysisCPUGPU2016. Unfortunately convolution reverbs do not expose many parameters or controls, making labeling of samples which are fed to our loss network (see @loss_network) difficult.

Parameter based reverberation, like delay networks are fast and require little memory, but careful tuning is necessary to find configurations that sound realistic @schlechtFeedbackDelayNetworks2018 @siddiqOptimizationConvolutionReverberation2020. This gives us easy access to e.g. size and wetness controls that we can use for labeling (see @loss_network).

In computational acoustics room simulations are used for reverberating audio @lemercierStoRMDiffusionbasedStochastic2023. Game engines such as Unity or libraries like pyroomacoustics can be used to simulate rooms with different sizes, materials and microphone placements. This is done by either by trying to solve the wave-equation by the discretization of the space, geometric solutions like the Image Source Method (ISM) @allenImageMethodEfficiently1979 or ray tracing @vorlanderAuralizationFundamentalsAcoustics2008. While this is attempting to recreate an acoustic space as close as possible, it is also the most computationally expensive and not possible to do live or offline for our amount of data. Unitys processing is also done in realtime, which makes it not feasable, as the runtime would be about 324 hours (cf. @dataset_split).

A first implementation was done using live... #TODO[]

- two kinds of preprocessing either live during training in memory saving on disk space or before "offline" saving on compute during training but sacificing disk space
  - first implementation was in memory
    - both reverberation through RIRs and Parameter reverb were implemented
    - using this the original conv tasnet dataloader was adjusted to use the in memory RIR implementation
  - after getting access to online compute (with TB+ space) we ditched in memory approach as training compute time was of more importance
  - VST based parameter reverb is os dependent and make for a horrible workflow

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


- sample rate: upscaling downscaling possible??
- short usability study what sampling (higher limit) rates are possible in real world scenarios (DAC)

- AudioSet is 44.1kHz: 10790 files
  - theoretically not fully DRY audio
- LibriMix/LibriSpeech is 16kHz: 51232 files
- Freesound is 44.1kHz: 46753 files
  - theoretically not fully DRY audio

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model

- saved precomputed values for :
  - "size": np.interp(size, SIZE_RANGE, [0, 1]), #sym.arrow schon normiert
  - "wetness": np.interp(wet, WET_RANGE, [0, 1]), #sym.arrow schon normiert

- live implementation as well as rir implementation for training of conv tasnet
- reverb done with parameter reverb
  - from pedalboard (FreeVerb implementation) @smithPhysicalAudioSignal2010

==== PEAQ

- offline
  - saved precomputed values for :
    - "odg": odg, #sym.arrow nicht normiert
    - "di": di, #sym.arrow nicht normiert


- make data
  - peaq (what implementation was used, namedrop authors for credebility)
  - reverberation techniques @smithPhysicalAudioSignal2010
  - upsampling to 44100 (and 48000 for peaq)
  - used parameter reverb because of better size and wetness control


==== Non-Silent Parts

- Ratio of total sample duration to non silent parts: prob about 70%
-> meaning that 30% of the time (excluding utterances that needed zero padding to get to our desired 2 or 4 second segment length) the model would train on pure silence. Therefore we needed to mask the silent and zero padded parts to lessen their impact when calculating loss.
To stop the model from learning to generate silence

#figure(
  caption: [Mask],
  image("/experiments/perceptual-quality/plots/mask_plot.svg"),
)

Duration of all train samples combined: 819907050.9525146 ms
Duration of all non silent utterances in training data: 539188097 ms
Ratio of non silent duration to full duration: 0.6576210027387939

== LOSS<loss_network>
#jojo
- why nn as loss (better score for perceptual, combines perceptual and "real world" attribs)
- why mel scale not bark etc.
go through loss network and explain weights (quality, size, wetness, odg) etc. make links to how data was processed for this task

- cite similar papers in zotero loss subcollection (like LEAN, etc.) for fast audio classification
  - why our loss model was based on CNN14
  - runtime (inference) evaluation

- general comparison of different loss functions in audio ML (sisnr, pesq, mse, l1, our own)

#figure(
  caption: [Metrics usable as loss functions analysed over 6236 datapoints from test dataset, outliers removed (data between 15th and 85th percentile)],
  image("/experiments/perceptual-quality/plots/data_metrics_test_6236_15_85_percentile.svg"),
)

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
