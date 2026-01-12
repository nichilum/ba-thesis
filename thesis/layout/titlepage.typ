#import "/layout/fonts.typ": *
#import "/layout/titlepage_table.typ": render-title-table

#let titlepage(
  title: "",
  titleGerman: "",
  degree: "",
  program: "",
  examiner: "",
  supervisors: (),
  authors: "",
  startDate: datetime,
  submissionDate: datetime,
) = {
  // Quality checks
  assert(degree in ("Bachelor", "Master"), message: "The degree must be either 'Bachelor' or 'Master'")
  
  set page(
    margin: (left: 20mm, right: 20mm, top: 30mm, bottom: 30mm),
    numbering: none,
    number-align: center,
  )

  set text(
    font: fonts.body, 
    size: 12pt, 
    lang: "en"
  )

  set par(leading: 0.5em)

  
  // --- Title Page ---
  v(1cm)
  align(center, image("/figures/th_logo.svg", width: 26%))

  v(5mm)
  align(center, text(font: fonts.sans, 2em, weight: 700, "Cologne University of \nApplied Sciences"))

  v(5mm)
  align(center, text(font: fonts.sans, 1.5em, weight: 100, "Faculty of Information, Media and Electrical Engineering"))
  
  v(15mm)

  align(center, text(font: fonts.sans, 1.3em, weight: 100, degree + "’s Thesis in " + program))
  v(8mm)
  

  align(center, text(font: fonts.sans, 2em, weight: 700, title))
  

  align(center, text(font: fonts.sans, 2em, weight: 500, titleGerman))

  let entries = ()
  
  if authors.len() > 0 {
    let authorField = "Author" + if authors.len() > 1 [s]
    entries.push((authorField, authors.join(", ")))
  }
  
  entries.push(("Examiner", examiner))
  // Only show supervisors if there are any
  if supervisors.len() > 0 {
    let supervisorField = "Supervisor" + if supervisors.len() > 1 [s]
    entries.push((supervisorField, supervisors.join(", ")))
  }
  entries.push(("Start Date", startDate.display("[day].[month].[year]")))
  entries.push(("Submission Date", submissionDate.display("[day].[month].[year]")))

  v(1cm)
  render-title-table(entries)
}
