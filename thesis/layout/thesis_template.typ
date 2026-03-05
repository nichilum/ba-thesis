#import "/thesis/layout/cover.typ": *
#import "/thesis/layout/titlepage.typ": *
#import "/thesis/layout/disclaimer.typ": *
#import "/thesis/layout/acknowledgement.typ": acknowledgement as acknowledgement_layout
#import "/thesis/layout/transparency_ai_tools.typ": transparency_ai_tools as transparency_ai_tools_layout
#import "/thesis/layout/abstract.typ": *
#import "/thesis/utils/print_page_break.typ": *
#import "/thesis/layout/fonts.typ": *
#import "/thesis/utils/diagram.typ": in-outline

#import "@preview/glossy:0.9.0": *

#let thesis(
  title: "",
  titleGerman: "",
  degree: "",
  program: "",
  examiner: "",
  supervisors: (),
  author: "",
  startDate: datetime,
  submissionDate: datetime,
  abstract_en: "",
  abstract_de: "",
  acknowledgement: "",
  transparency_ai_tools: "",
  is_print: false,
  body,
) = {
  // cover(
  //   title: title,
  //   degree: degree,
  //   program: program,
  //   author: author,
  // )

  // pagebreak()

  let myGlossary = (
    // html: (
    //   short: "HTML",
    //   long: "Hypertext Markup Language",
    //   description: "A standard language for creating web pages",
    //   group: "Web",
    // ),
    // css: (
    //   short: "CSS",
    //   long: "Cascading Style Sheets",
    //   description: "A stylesheet language used for describing the presentation of a document",
    //   group: "Web",
    // ),
    // tps: (
    //   short: "TPS",
    //   long: "test procedure specification",
    //   description: "A formal document describing test steps and expected results",
    //   // Optional: Override automatic pluralization
    //   plural: "TPSes",
    //   longplural: "test procedure specifications",
    // ),
    RIR: "Room impulse response",
  )

  show: init-glossary.with(myGlossary)

  titlepage(
    title: title,
    titleGerman: titleGerman,
    degree: degree,
    program: program,
    examiner: examiner,
    supervisors: supervisors,
    authors: author,
    startDate: startDate,
    submissionDate: submissionDate,
  )

  print_page_break(print: is_print, to: "even")

  disclaimer(
    title: title,
    degree: degree,
    author: author,
    submissionDate: submissionDate,
  )
  transparency_ai_tools_layout(transparency_ai_tools)

  print_page_break(print: is_print)

  acknowledgement_layout(acknowledgement)

  print_page_break(print: is_print)

  abstract(lang: "en")[#abstract_en]
  abstract(lang: "de")[#abstract_de]

  set page(
    margin: (left: 30mm, right: 30mm, top: 40mm, bottom: 40mm),
    numbering: none,
    number-align: center,
  )

  set text(
    font: fonts.body,
    size: 12pt,
    lang: "en",
  )

  show math.equation: set text(weight: 400)

  // --- Headings ---
  show heading: set block(below: 0.85em, above: 1.75em)
  show heading: set text(font: fonts.body)
  set heading(numbering: "1.1")
  // Reference first-level headings as "chapters"
  show ref: it => {
    let el = it.element
    if el != none and el.func() == heading and el.level == 1 {
      link(
        el.location(),
        [Chapter #numbering(el.numbering, ..counter(heading).at(el.location()))],
      )
    } else {
      it
    }
  }

  // --- Tables ---
  set table(stroke: (_, y) => if y > 0 { (top: 0.8pt) } else if y == 0 { (bottom: 0.8pt) }, row-gutter: (2.2pt, auto))

  // --- Paragraphs ---
  set par(leading: 1em)

  // --- Citations ---
  set cite(style: "alphanumeric")

  // --- Figures ---
  show figure: set text(size: 0.85em)

  // --- Table of Contents ---
  show outline.entry.where(level: 1): it => {
    v(15pt, weak: true)
    strong(it)
  }
  outline(
    title: {
      text(font: fonts.body, 1.5em, weight: 700, "Contents")
      v(15mm)
    },
    indent: 2em,
  )


  v(2.4fr)
  pagebreak()

  let theme-a = (
    section: (title, body) => {
      heading(numbering: none, title)
      v(1em)
      body
    },
    group: (name, index, total, body) => {
      if name != "" and total > 1 {
        v(1.5em)
        align(left, text(weight: "bold", size: 1.2em, name))
        v(0.75em)
        line(length: 100%, stroke: 0.5pt)
        v(0.75em)
      }
      body
    },
    entry: (entry, index, total) => {
      let short-display = text(weight: "bold", entry.short)
      let long-display = if entry.long == none {
        []
      } else {
        [. #entry.long]
      }

      let description = if entry.description == none {
        []
      } else {
        [. #entry.description]
      }

      block(
        below: 1em,
        text(
          size: 0.95em,
          {
            grid(
              columns: (1fr, auto),
              gutter: 0.75em,
              [#short-display#long-display#description#entry.label], text(fill: rgb("#666666"), entry.pages.join(", ")),
            )
          },
        ),
      )
    },
  )

  glossary(
    title: "Glossary", // Optional: defaults to Glossary theme:
    theme: theme-a,
    sort: true, // Optional: whether or not to sort the glossary
    ignore-case: false, // Optional: ignore case when sorting terms
    show-all: false, // Optional; Show all terms even if unreferenced
  )
  pagebreak()


  // Main body. Reset page numbering.
  set page(numbering: "1")
  counter(page).update(1)
  set par(justify: true, first-line-indent: 2em)

  body

  // List of figures.
  pagebreak()
  heading(numbering: none)[List of Figures]
  show outline: it => {
    // Show only the short caption here
    in-outline.update(true)
    it
    in-outline.update(false)
  }
  outline(
    title: "",
    target: figure.where(kind: image),
  )

  // List of tables.
  context [
    #if query(figure.where(kind: table)).len() > 0 {
      pagebreak()
      heading(numbering: none)[List of Tables]
      outline(
        title: "",
        target: figure.where(kind: table),
      )
    }
  ]

  // Appendix.
  pagebreak()
  heading(numbering: none)[Appendix A: Supplementary Material]
  include "/thesis/layout/appendix.typ"

  pagebreak()
  bibliography("/thesis/dereverberation.bib")
}
