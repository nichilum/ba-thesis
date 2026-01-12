#import "/utils/todo.typ": TODO

= Introduction

- daten vorbereitung/sammeln
  - datensätze und/oder selber aufnehmen
  - raumsimulation 
- dereverberation
  - tasnet bzw. conv-tasnet
    - ist tasnet gut für uns? (weil auch musik)
  - zusätzlich:
    - eigenes netz in time domain trainieren (orientieren an tasnet arch)
    - eigenes netz in freq domain trainieren (orientieren an Ernst et. al.)
    - vergleich in nicht real time -> welcher ansatz ist der qualitativ bessere
  - \<50ms real time
  - welche sampling rate 
    - höhere sampling rate besser???? 
    - ist bei einem convolution netzwerk mit frequency-domain ansatz eine höhere auflösung besser???? 
    - näher an real time dran durch z.B. 96 kHz (bei 96kHz und 5ms time window 200 Hz pro bin -> relativ hochauflösend)
  - evaluation?: "plausibilität", wie "gut" klingt es, rein subjektives maß
    - Vergleich von input (dry) zu ouput: coloration, noise, etc.
- optional: real world test
  - kleiner Aufbau, ein/zwei Mikros plus Lautsprecher
- theoretisch anpassbar auf Anwendung: feedback bereinigung


// #TODO[ // Remove this block
//   *Introduction*
//   - Introduce the reader to the general setting (No Problem description yet)
//   - What is the environment?
//   - What are the tools in use?
//   - (Not more than 1/2 a page)
// ]
