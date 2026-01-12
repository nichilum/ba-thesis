#import "/layout/proposal_template.typ": *
#import "/metadata.typ": *
#import "/utils/todo.typ": *

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
  transparency_ai_tools: include "/content/proposal/transparency_ai_tools.typ",
)

#set heading(numbering: none)
// #include "/content/proposal/abstract.typ"

#set heading(numbering: "1.1")
#include "/content/proposal/introduction.typ"
#pagebreak()
#include "/content/proposal/problem.typ"
#pagebreak()
#include "/content/proposal/motivation.typ"
#pagebreak()
#include "/content/proposal/objective.typ"
#pagebreak()
#include "/content/proposal/schedule.typ"
#pagebreak()
