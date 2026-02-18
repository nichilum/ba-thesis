#import "/thesis/layout/thesis_template.typ": *
#import "/thesis/metadata.typ": *

#set document(title: titleEnglish, author: authors)

#show: thesis.with(
  title: titleEnglish,
  titleGerman: titleGerman,
  degree: degree,
  program: program,
  examiner: examiner,
  supervisors: supervisors,
  author: authors,
  startDate: startDate,
  submissionDate: submissionDate,
  abstract_en: include "/thesis/content/abstract_en.typ",
  abstract_de: include "/thesis/content/abstract_de.typ",
  acknowledgement: include "/thesis/content/acknowledgement.typ",
  transparency_ai_tools: include "/thesis/content/transparency_ai_tools.typ",
)

#include "/thesis/content/introduction.typ"
#include "/thesis/content/background.typ"
#include "/thesis/content/related_work.typ"

#include "/thesis/content/fundamentals.typ"
#include "/thesis/content/methodology.typ"
#include "/thesis/content/experiments.typ"

#include "/thesis/content/results.typ"
#include "/thesis/content/evaluation.typ"
#include "/thesis/content/conclusion.typ"
