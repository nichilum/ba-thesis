#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ

= Implementation & Experimental Setup

== Conv-TasNet for diverse audio dereverberation
- no proper weights for training available
  - I forgot the proper reasoning why we did not use weights available on huggingsface and google drive (linked on one github repo)
    - only for speaker seperation
    - reached out to original authors but did not get a response
- used model implementation by the author linked in the original paper @luoConvTasNetSurpassingIdeal2019
  - trained using LibriMix dataset @panayotovLibrispeechASRCorpus2015, original WSJ0-2mix and WSJ0-3mix datasets @garofolojohns.CSRIWSJ0Complete2007 are not publically available
  - original loss function (SI-SNR) only resulted in no convergence (stayed negative) and thus unusable results
  - switched SI-SNR loss to MSE loss which resulted in convergence and usable results
  - show training and validation loss plots
  - show some example predictions (spectrograms and audio)

== Own Implementation
- it was shown that modifying the Conv TasNet TCN based architecture for a fully generative approach (no mask, but generate the final audio from the TCN representation) is not feasable with low computational cost (overfittable but doesn't generalize well)
  - show plots


*inverse estimation in encoder space*
- frequency
- time (ConvTasNet)
  - use conv tasnet mask
  - test for diverse audio signals
  - compare mse to perceptual loss


#figure(
  caption: [Prediction quality of perceptual net from signal with increasing zero percentage],
  image("/experiments/perceptual-quality/plots/perceptual_net_zeros_preds.svg"),
)

#TODO[https://typst.app/universe/package/neural-netz]
