#import "/layout/fonts.typ": *

#let disclaimer(
  title: "",
  degree: "",
  author: "",
  submissionDate: datetime,
) = {
  set page(
    margin: (left: 30mm, right: 30mm, top: 40mm, bottom: 40mm),
    numbering: none,
    number-align: center,
  )

  set text(
    font: fonts.body, 
    size: 12pt, 
    lang: "en"
  )

  set par(leading: 1em)

  
  // --- Disclaimer ---  
  v(75%)
  text("We hereby confirm that this " + degree + "’s thesis is our own work and we have documented all sources and material used.")

  v(15mm)
  grid(
      columns: 2,
      gutter: 1fr,
      "Cologne, " + submissionDate.display("[day].[month].[year]"), author.join(" & ")
  )
}
