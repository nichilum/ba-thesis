#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ

= Methodology

== Dataset

=== Data Collection

Dies das irgendwas mit Unity schreiben, dass das auch eine Möglichkeit gewesen wäre Räume zu simulieren, aber zu lange gedauert hätte, da simulation nur in realtime, bei unser menge an daten (k.a. 500+ stunden), maybe paar screenshots

- AudioSet
- LibriSpeech (LibriMix)
- Freesound
- RiR for TASNet training
\
- total length, what classes are covered
- look at PANNs paper

=== Data Preprocessing

- sample rate: upscaling downscaling possible??
- short usability study what sampling (higher limit) rates are possible in real world scenarios (DAC)
- make data
  - peaq (what implementation was used, namedrop authors for credebility)
  - reverberation techniques
  - upsampling to 44100 (and 48000 for peaq)
  - used parameter reverb because of better size and wetness control

== LOSS
- why nn as loss (better score for perceptual, combines perceptual and "real world" attribs)
- why mel scale not bark etc.
go through loss network and explain weights (quality, size, wetness, odg) etc. make links to how data was processed for this task

- cite similar papers in zotero loss subcollection (like LEAN, etc.) for fast audio classification
  - why our loss model was based on CNN14
  - runtime (inference) evaluation

- general comparison of different loss functions in audio ML (sisnr, pesq, mse, l1, our own)

#figure(
  caption: [Metrics usable as loss functions analysed over 5000 datapoints from training dataset],
  image("/experiments/perceptual-quality/plots/data_metrics_5000.svg"),
)

Key takeaways:
- wetness and size are objective measurements which we know to be true: $lim_("wet"arrow 1)$ and $lim_("size"arrow 1)$ means the signal is badly reverberated and $lim_("wet"arrow 0)$ and $lim_("size"arrow 0)$ means the signal is dereverberated
- correlation, mae and mse are bad loss functions as they do not accurately predict wetness or size values
- odg shows more "bad" (close to 0) values around higher wetness or size values, which is what we "need" from a loss function
- di does it similarily but we cannot normalize it that well
- si snr could also be used but experiments with tasNet showed even it inferior or close to just the standard mse
- train network on combination of odg, size and wetness resulting in quality score (lowest graph), which accurately predicts size and wetness

quality is here defined as:
$ Q = "ODG" dot (1 - "wet" dot 0.4) dot (1 - "size" dot 0.3) $

- plot is little pointless here: akin to plotting wetness and size against theirselfs, BUT in the end this quality function will be estimated using Neural Network


TODO: redo plots using test data and also add quality NETWORK graph with predictions from NN
