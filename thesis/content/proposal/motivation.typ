#import "/utils/todo.typ": TODO

= Motivation

// Reverberation is a fundamental property of sound, but excessive or uncontrolled reverberation can significantly degrade the quality and intelligibility of speech, music, and environmental recordings. Many existing datasets lack the acoustic diversity or realism needed to evaluate modern dereverberation methods, making data collection—either through curated datasets, custom recordings, or room simulations—a crucial foundation for developing reliable algorithms. As machine-learning-based dereverberation has advanced rapidly in recent years, there is an opportunity to investigate how different model architectures perform under controlled but realistic acoustic conditions, and how data choice, simulation fidelity, and sampling rate influence model behavior.

As machine-learning based dereverberation has advanced in recent years, works such as Conv-TasNet have shown remarkable effectiveness in speech separation and dereverberation, yet their suitability for complex signals such as music remains unclear @luoConvTasNetSurpassingIdeal2019. At the same time, alternative architectures—both in the time domain and frequency domain—offer theoretical advantages but lack direct, systematic comparison @luoTasNetTimedomainAudio2018 @ernstSpeechDereverberationUsing2018 @luoRealtimeSinglechannelDereverberation2018. This thesis is motivated by the need to understand which approaches yield the highest perceptual and quantitative quality when real-time constraints are taken into account. Exploring the impact of sampling rate, spectral resolution, and model design we aim to provide valuable insights into how dereverberation systems can be optimized for general purpose use.
  
// #TODO[ // Remove this block
//   *Proposal Motivation*
//   - Outline why it is (scientifically) important to solve the problem
//   - Again use the actors to present your solution, but don't be to specific
//   - Do not repeat the problem, instead focus on the positive aspects when the solution to the problem is available
//   - Be visionary! 
//   - Optional: motivate with existing research, previous work 
// ]
