#import "/utils/todo.typ": TODO
#import "../../utils/open_questions.typ": OPENQ
#import "@preview/gantty:0.5.1": gantt

= Schedule

== Work Packages

#let wp(title, owner, body) = [
=== Work Package: *#title*
*Responsible:* #owner

#body
]


== Gantt Chart
#set page(flipped: true)
#gantt(yaml("gantt.yaml"))
#set page(flipped: false)




// #TODO[ // Remove this block
//   *Thesis Schedule*
//   - When will the thesis Start
//   - Create a rough plan for your thesis (separate the time in iterations with a length of 2-4 weeks)
//   - Each iteration should contain several smaller work items - Again keep it high-level and make to keep your plan realistic
//   - Make sure the work-items are measurable and deliverable, they should describe features that are vertically integrated
//   // - Do not include thesis writing or presentation tasks
// ]
