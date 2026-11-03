# Location audit rubric: where does the wrong solution first go wrong? (v1, 2026-11-03)

Each item is a WRONG solution of Qwen2.5-1.5B-Instruct to a GSM8K problem. Every equation that can be checked is
marked right after its result: `[E1]`, `[E2]`, ... Find where the solution FIRST goes wrong. Use the GSM8K reference
solution and, if shown, a correct solution by the same model to see what should happen. Other valid approaches are
fine.

## Fields

`id<TAB>first<TAB>words<TAB>prose<TAB>conf<TAB>reason`

- **first**: the first marked equation whose result is a WRONG value: `E1`, `E2`, ...; `none` if every marked result
  is fine.
  - WRONG: computed incorrectly; or a number for the wrong quantity, or built on a misunderstanding of the problem, so
    it cannot be part of a correct solution.
  - FINE: correct and part of some valid solution path (also a different approach than the reference); or a correct
    side computation that is unnecessary but harmless.
  - A result that is wrong only because an earlier result was wrong is not the first one: pick the earliest.
- **words**: `y` if, before that equation (anywhere, if first = `none`), the text already contains a wrong number that
  is NOT in a marked equation, e.g. "180 minutes is 1 hour 30 minutes", "so she has 12 apples" (should be 10),
  "2% of 750 is 150". Otherwise `n`.
- **prose**: `y` if, before the first wrong number, the text already states a misunderstanding in words, with no wrong
  number yet, e.g. "1/3 of the cake was eaten" (the problem says 1/3 is left), "each team has one coach" (one per
  school). Otherwise `n`.
- **conf**: `h` clear; `m` another choice for `first` is plausible; then write `alt=E#` (or `alt=none`) in the reason.
- **reason**: at most 15 English words: what is wrong at `first` (or where the error is, if `none`).

## Rules

1. Judge the values, not the style: unit words, rounding to the shown precision and restating an earlier result are
   fine.
2. If the model uses a different but valid method, its intermediate values are fine.
3. If every marked equation is fine and the error is only in words or at the final answer, write `first = none` and
   say where the error is in the reason.
4. Judge the math. Do not try to guess what an automatic checker would pick.
