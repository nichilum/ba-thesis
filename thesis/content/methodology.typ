#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ
#import "/thesis/utils/author.typ": *

= Methodology

== Dataset

Other machine learning fields mainly computer vision (CV) and large language models (LLMs) have long been trained on publically available diverse datasets @dengImageNetLargeScaleHierarchical2009 namely mC4, MassiveText or the Wikipedia dataset @naveedComprehensiveOverviewLarge2025.

As shown in @related_work previous work in the field of audio dereverberation has generally focused on speech signals. As this limitation is the same for many audio based machine learning problems (e.g. multi speaker seperation, noise cancellation and speech to text) many of the most used large audio datasets consist only of speech signals which are reduced in bandwidth as well as language diversity and recored in anechoic conditions @garofolojohns.CSRIWSJ0Complete2007 @panayotovLibrispeechASRCorpus2015 @richterEARSAnechoicFullband2024.

Datasets of diverse audio signals have emerged from audio classification problems. Early examples being private self collected datasets of indivdual researchers @woodardModelingClassificationNatural1992 @ellisDetectingAlarmSounds2001.
Over the recent years interest in audio classification has surged as can be seen in the amount of entries in the "Detection and Classification of Acoustic Scenes and Events" (DCASE) challenge series that increased from 31 in 2013 to 428 in 2023 @mesarosDecadeDCASEAchievements2024. The DCASE has also been a major influence in the increase of publically available datasets as prior to the DCASE challenges only a limited amount were available most notably RWCP @smithPhysicalAudioSignal2010.

The current largest dataset of diverse audio signals is Google's fittingly named AudioSet containing over 5,800 hours of audio recordings with 527 classes
of annotated sounds @gemmekeAudioSetOntology2017. These recordings are 10 second clips drawn from YouTube videos. Building on top of the AudioSet classes the FSD50K dataset contains 100 hours of audio composed of 51,197 individual samples @fonsecaFSD50KOpenDataset2022 taken from the "freesound.org" audio sharing site. The FSD50K dataset is publically available while AudioSet released embedding features of the raw audio data necessitating a private download from YouTube. Both datasets are human-labeled while AudioSet specifies that sounds are human-verified and classes are suggested using YouTube metadata.

=== Data Collection

Our proposed approach requires a diverse dataset of dry audio data. In total 108,775 indivdual audio samples were collected resulting in the following dataset:

#figure(caption: [Dataset split], table(
  columns: 3,
  [*Dataset*], [*Number of Files*], [*Length of Files*],
  [AudioSet], [10790 (9.92 %)], [],
  [FSD50K], [46753 (42.98 %)], [],
  [LibriSpeech/LibriMix], [51232 (47.1 %)], [],
))

Diverse audio data from the AudioSet and FSD50K datasets were downloaded in 44.1 kHz. Both datasets were used as to eliminate any bias occurring in one of the datasets (e.g. YouTube compression artifacts). The LibriSpeech dataset includes english utterances recored in anechoic conditions and samples at 16 kHz. These were included in hopes of giving speech signals a greater weight as we felt clean speech was underrepresented in the other datasets.

// - RiR for TASNet training @jeub09a
// - what classes are covered
//   - look at PANNs paper for AudioSet Citation
// - own downloader, scraped from youtube in 44.1 kHz, talk about tech used and the theoretical quality possible
//   - IN THEORY ILLEGAL: DMCA 1201 / (Urheberrechtsgesetz) § 95a Schutz technischer Maßnahmen
// - talk about size and what we managed to download

=== Data Preprocessing

Dies das irgendwas mit Unity schreiben, dass das auch eine Möglichkeit gewesen wäre Räume zu simulieren, aber zu lange gedauert hätte, da simulation nur in realtime, bei unser menge an daten (k.a. 500+ stunden), maybe paar screenshots

- sample rate: upscaling downscaling possible??
- short usability study what sampling (higher limit) rates are possible in real world scenarios (DAC)
- make data
  - peaq (what implementation was used, namedrop authors for credebility)
  - reverberation techniques @smithPhysicalAudioSignal2010
  - upsampling to 44100 (and 48000 for peaq)
  - used parameter reverb because of better size and wetness control


Duration of all train samples combined: 819907050.9525146 ms
Duration of all non silent utterances in training data: 539188097 ms
Ratio of non silent duration to full duration: 0.6576210027387939

*PROBLEMS*:
- AudioSet is 44.1kHz: 10790 files
  - theoretically not fully DRY audio
- LibriMix/LibriSpeech is 16kHz: 51232 files
- Freesound is 44.1kHz: 46753 files
  - theoretically not fully DRY audio

reverberation was made in native sample rate, then upsampled for training:
meaning that some files lack proper wide band reverberation and might "confuse" model



- Ratio of total sample duration to non silent parts: prob about 70%
-> meaning that 30% of the time (excluding utterances that needed zero padding to get to our desired 2 or 4 second segment length) the model would train on pure silence. Therefore we needed to mask the silent and zero padded parts to lessen their impact when calculating loss.
To stop the model from learning to generate silence


*REVERBERATION*:

- reverb done with parameter reverb
  - from pedalboard (FreeVerb implementation) @smithPhysicalAudioSignal2010
- offline
  - saved precomputed values for :
    - "size": np.interp(size, SIZE_RANGE, [0, 1]), #sym.arrow schon normiert
    - "wetness": np.interp(wet, WET_RANGE, [0, 1]), #sym.arrow schon normiert
    - "odg": odg, #sym.arrow nicht normiert
    - "di": di, #sym.arrow nicht normiert
- live implementation as well as rir implementation for training of conv tasnet


#figure(
  caption: [Mask],
  image("/experiments/perceptual-quality/plots/mask_plot.svg"),
)

== LOSS
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
