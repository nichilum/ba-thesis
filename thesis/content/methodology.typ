#import "/utils/todo.typ": TODO
#import "/utils/open_questions.typ": OPENQ

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
