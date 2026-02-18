#import "/thesis/layout/proposal_template.typ": *
#import "/thesis/metadata.typ": *
#import "/thesis/utils/todo.typ": *

#set document(title: titleEnglish, author: authors)

#show: proposal.with(
  title: titleEnglish,
  titleGerman: titleGerman,
  degree: degree,
  program: program,
  examiner: examiner,
  supervisors: supervisors,
  authors: authors,
  startDate: startDate,
  submissionDate: submissionDate,
  transparency_ai_tools: include "/thesis/content/proposal/transparency_ai_tools.typ",
)

#set heading(numbering: none)
// #include "/thesis/content/proposal/abstract.typ"

#set heading(numbering: "1.1")
#include "/thesis/content/proposal/introduction.typ"
#pagebreak()
#include "/thesis/content/proposal/problem.typ"
#pagebreak()
#include "/thesis/content/proposal/motivation.typ"
#pagebreak()
#include "/thesis/content/proposal/objective.typ"
#pagebreak()
#include "/thesis/content/proposal/schedule.typ"
#pagebreak()
