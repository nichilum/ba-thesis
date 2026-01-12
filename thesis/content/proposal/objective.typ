#import "/utils/todo.typ": TODO
#import "../../utils/open_questions.typ": OPENQ

= Objective

/ 1. Dataset: Develop and collect dataset using room simulation
/ 2. SoTA-Evaluation: Evaluate state of the art model for general purpose dereverberation (non-speech signals)
/ 3. Domain-Evaluation: Develop/customize and train time- and frequency-based architectures with similar latency for non-speech signals
  - Evaluate both architectures with respect to coloration/performance
/ 4. Incremental Learning: Investigate model with incremental learning capabilities (adjust to current live scenario)
  - Ensure reactivity on changing live scenarios


== Dataset
To evaluate and train our and other models, we will create a dateset consisting of various audio signals (speech and non-speech). A room simulation is used for the generation of dry–wet data pairs enabling self-supervised learning.

== SoTA-Evaluation
To create a baseline for the following evaluations we implement a state of the art model, possibly TasNet or Conv-TasNet. It is yet to be seen how well TasNet handles non-speech signals. The main objective is to investigate to which extend this is possible.

== Domain-Evaluation
Develop, customize, and train time-domain and frequency-domain network architectures for dereverberation of non-speech signals under comparable latency constraints. Evaluate both approaches with respect to perceptual quality and coloration artifacts. Analyze the influence of model parameters and sampling rate, particularly for frequency-domain methods, on dereverberation performance and real-time feasibility.

// - Combine multiple network architectures
// - combine multiple approaches
// - adjust parameters
// - all for non-speech signals
// - frequency based approach evaluate for different sampling rates (our idea: higher = better)

== Incremental Learning
Develop a discriminator-based model that evaluates the quality of dereverberated signals using features derived from similar network architectures. Use the resulting quality score as a feedback signal for incremental learning, enabling the system to adapt to changing acoustic conditions in live scenarios. Evaluate the reactivity and stability of the model with respect to dynamic environmental changes.

// - Develop discriminator based on similiar model architecture as above
//   - score dereverbed dry signal on how good dereverberation worked
//   - score can be any type (similar to encode decoder: latent space score, idk)
//   - score can be used as discriminator for incremental learning (kind of like a reactive filter adjustment)
    
// #TODO[ // Remove this block
//   *Proposal Objective*
//   - Define the main goals of your thesis clearly and concisely.
//   - Start with a short overview where you enumerate the goals as bullet points, using action-oriented phrasing (e.g., 1., 2., 3., ...).
//   - Avoid the gerund form for verbs (e.g., "Developing Feature XYZ") and noun phrases (e.g., "Feature XYZ Development"). Instead, use action-oriented language such as "Develop Feature XYZ", similar to how you would formulate use cases in UML use case diagrams.
//   - Ensure your goals are concrete and specific, avoiding generic statements. Clearly state what you aim to achieve.
//   - Expand on each goal in a dedicated subsection. Repeat the corresponding enumerated bullet point number to maintain consistency and provide at least two paragraphs explaining the goal. Focus on being precise and specific in your descriptions.
// ]
