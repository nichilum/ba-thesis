#import "/thesis/utils/todo.typ": TODO
#import "/thesis/utils/open_questions.typ": OPENQ

= Implementation & Experimental Setup



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
