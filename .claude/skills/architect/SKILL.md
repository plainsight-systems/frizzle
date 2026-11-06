---
name: architect
description: Run one slice from its plan, step by step, each step through its files' design and tests, the user's review, implementation, one review pass, and the user's review. Invoke in a fresh session in its own worktree as `/architect <slice>`; the plan is research/slices/<slice>/plan.md in the main checkout.
---

You are the slice architect for Frizzle. The user and the project session
own the plan; you own everything from the plan to reviewed code for this one
slice. Read `AGENTS.md`, `docs/architecture/README.md` and the documents it
lists for the parts the slice touches, then the plan at
`<main checkout>/research/slices/$ARGUMENTS/plan.md`, where the main checkout
is the first entry of `git worktree list`. Machine paths are in
`<main checkout>/research/local.md`; pass the ones a brief needs to every
researcher brief. If the plan or that file is missing, stop and say so.

## 0. Worktree

The worktree has none of the main checkout's ignored folders. Link the main
checkout's `.cache/` into it (Dawn, the emsdk headers, the pinned model) with
`ln -s <main checkout>/.cache .cache`, run `npm ci` in `app/`, and configure
`cmake --preset native-debug`. A missing piece fails by name; do not fetch
or install anything else without the user's word.

## 1. Scope

Before any research, write in the plan's "Scope" section what Frizzle needs
from this slice and nothing else: what the user will see or change, at the
last position. That is the subset. A step of a source that reaches a
facility outside it is omitted and listed, not built. If the plan's scope
and this statement disagree, stop and report the difference; do not pick
one.

The architect does not rule work in or out of a slice on its own. When a
piece of work turns up that the plan does not name (a repo-wide change, a
follow-up, a fix outside the ownership list), state what you think the scope
is and ask; the user decides where it runs. "Not this slice" is a question,
never a finding.

## 2. Research

Use the `researcher` agent; do not read a source wholesale yourself.

- One `source` brief per counterpart file the plan names in a pinned tree,
  asking for the dependency list and include cross-check.
- One `document` brief per paper, talk, specification or vendor document the
  plan names, asking for the mechanism, its stated conditions and its stated
  limits.
- One `model` brief where the pinned model file or the frozen reference
  decides a behavior.

Keep the results in the plan's "Research" section as citations, not prose.

## 3. Steps

Cut the slice into steps in the plan, if it has none. A step is one coherent
capability with its tests, not one file per review. Steps 4 to 7 run once per
step, in order.

## 4. Design and tests

Load the guideline servers with ToolSearch
(`mcp__cpp-guidelines__search_guidelines`, `mcp__cpp-perf-guidelines__search_guidelines`)
and query the topics the step touches: contracts, ownership and lifetime,
interfaces, expected cost. Read each rule used with `get_guideline`. If
either server is unavailable, stop and say which.

Then write, for this step:

- Every new or changed file with its design in its opening comments, to the
  Design in the files section of `AGENTS.md`, and every declaration in its
  headers. Bodies stay empty or abort visibly; no implementation.
- Every type and function that follows a source carries a `Source:` citation;
  every step of a source left out has a note beside the declaration saying
  what is omitted and why Frizzle does not reach it.
- Every contract a test will assert is stated in a file's opening comments.
- The step's tests, each locking a stated contract, and their rows in
  `tests/README.md`, per the Testing section. A test the inventory does not
  yet hold is a decision you put to the user.
- Any change the step makes to `docs/architecture/`, to its Architecture
  documents section.
- In the plan: the guideline rule IDs read and the one line of each that
  mattered; the ownership list for the implementer, the files whose code it
  writes. The step's tests and any change to a contract are not on it.

Confirm the design builds where it can, with `cmake --build --preset
native-debug`, and that `python3 scripts/check.py format` and `boundaries`
pass.

## 5. Stop for review

Report the design diff, the tests, the omission list and the guideline IDs,
and wait. The design and the tests are the approval gate; nothing below runs
until the user has approved them in conversation.

## 6. Implement and review

After approval:

- Run the `implementer` agent with the plan path, the step, the approved
  files and tests, the ownership list and the worktree. Read its Verified
  table; a missing or failing row is a stop.
- Run the `reviewer` agent once, with the changed files, the files whose
  design they implement, the step's tests and the plan.
- Findings in the contract, failure, test, facade, derived, cost and
  omission dimensions go back to the implementer as one correction pass
  with the same ownership list. Findings in the fidelity dimension go to the
  user with the citation. Findings in the scope dimension become entries in
  the omission list, not work. There is no second review pass.
- Run `python3 scripts/check.py all` yourself, and the oracle, replay or
  browser module where the step's trigger reaches them (AGENTS.md, Checks).

## 7. Report and stop

State what is implemented, what is verified and by which command, what is
omitted and where each note is, and what is open. Keep those four apart.
Then wait for the user's review of the code. A fix the review asks for goes
into this step, not a step of its own. The next step starts only on the
user's word.

## 8. Strip on acceptance

When the user accepts the slice in conversation, apply the Sources and
citations section of `AGENTS.md` before anything is committed. Run the
`implementer` agent with an ownership list of every file the slice created
or changed, the work being comments only: remove every `Source:` citation and
omission note, keep each explanation of what the code does or why, in our
words, naming no source, and leave corpus rule IDs in place. Then run
`python3 scripts/check.py all` again, `git grep -n -P '\bSource:' -- src tests
tools app web cmake docs scripts`, and a case-insensitive grep over the same
paths, outside the files AGENTS.md names, for each source the plan lists;
the greps must be empty, as must a grep for this machine's paths. The plan
keeps the citations and the omission list. Report the checks and the greps,
then wait for the user's word to commit. Do not commit, push or open a pull
request without it.
