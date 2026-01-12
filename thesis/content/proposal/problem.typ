#import "/utils/todo.typ": TODO
#import "../../utils/open_questions.typ": OPENQ

= Problem

Reverberation is apparent in every audio signal as it is an inherent characteristic of recording environments. It was shown that reverberation is an important auditory cue which informs the listener over environmental factors @traerStatisticsNaturalReverberation2016. Depending on the application reverberation can either be an attractive addition to the auditory signal, such as in music @NAYLOR2014879. While studies have shown that human speech recognition performs worse on reverberant signals @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021.

Artificial reverberation can be added to audio signals with comparatively simple signal processing techniques, the inverse task—removing or reducing existing reverberation—is significantly more complex @attiasSpeechDenoisingDereverberation2000. Reverberation is a time-dispersive and highly non-linear process, where direct sound and multiple delayed reflections overlap in both time and frequency. This overlap makes a clear separation between the original (dry) signal and the reverberant components difficult and, for a long time, was considered practically unsolvable using classical digital signal processing methods @brandsteinUseExplicitSpeech1998.

In many modern applications however, dereverberation is highly desirable. We divide use cases into two main categories: _offline_ and _live_ processing. Offline applications do not strictly require real-time operation, although real-time capability may still be beneficial. Typical examples include music remixes and film post-production, where excessive room reverberation can reduce audio clarity and limit creative flexibility. In contrast, live applications inherently require real-time processing, as they are used in interactive scenarios such as video conferencing, speech recognition systems, and live music performance. In these cases, reverberation can significantly degrade speech intelligibility, introduce unwanted coloration, and negatively affect the overall user experience @neumanCombinedEffectsNoise2010 @puglisiEffectReverberationNoise2021. As a result, live applications impose strict constraints on processing latency and computational efficiency. This thesis focuses on real-time solutions applicable to live applications, which could be transferred for use in offline scenarios.

This leads to several open challenges addressed in this thesis. First, it is unclear how well established dereverberation architectures generalize to diverse audio signals such as music and mixed content @luoTasNetTimedomainAudio2018.  Second, there are time-domain and frequency-domain approaches, which can be investigated in terms of audio quality, computational complexity, and latency. Third, real-time applicability imposes strict latency limits (e.g. below 50 ms) @schmidMeasuringJustNoticeable2024 that strongly influence network architecture, window size, and sampling rate. The impact of higher sampling rates on dereverberation performance and real-time feasibility is also not well understood.

The central problem of this thesis is therefore to investigate whether deep-learning-based dereverberation methods can be designed to operate in real time while maintaining perceptually convincing audio quality for a wide range of audio signals. This includes comparing time-domain and frequency-domain neural network approaches, evaluating their qualitative performance, and analyzing their suitability for low-latency, real-time applications.

// Adding reverb is easy, removing it is not.
// Long thought to be impossible, managable through machine learning and dsp.




// - offline (nicht umbedingt echtzeit fähig)
//   - z.B. Remixe (Musik generell)
//   - bereinigung von Aufnahmen (Film, etc.)
// - live (auf jeden fall echtzeitfähig fähig)
//   - video calls
//   - speech recognition
//   - live music (active acoustics)


  
// SOUND RECORDINGS:
// - https://www.epidemicsound.com/music/genres/opera/
// - https://pixabay.com/de/music/search/oper/
// - https://pixabay.com/music/search/orchestral/
// - https://www.rhenania.de/musik/klassik/
// - https://freemusicarchive.org/genre/Symphony/

// #TODO[ // Remove this block
//   *Problem description*
//   - What is/are the problem(s)? 
//   - Identify the actors and use these to describe how the problem negatively influences them.
//   - Do not present solutions or alternatives yet!
//   - Present the negative consequences in detail 
// ]
