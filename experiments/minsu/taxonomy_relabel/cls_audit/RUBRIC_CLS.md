# Class + location audit rubric: what kind of first error, and where? (v1, 2026-11-03)

Each item is a WRONG solution of Qwen2.5-1.5B-Instruct to a GSM8K problem (its final answer differs from the gold
answer). Every non-empty line of the wrong solution is numbered `[L1]`, `[L2]`, ... Use the GSM8K reference solution
and, if shown, a correct solution by the same model to see what should happen. Other valid approaches are fine.

## Fields

`id<TAB>class<TAB>num<TAB>thought<TAB>conf<TAB>reason`

- **class**: decided by the FIRST error in the solution.
  - `CALC`: the first wrong value is an arithmetic slip. The numbers and the operation written are the ones the
    solution meant, but the result is wrong, e.g. "45 * 3 = 125", "half of 36 is 16".
  - `TARGET`: every value the solution computes is correct (part of some valid solution path), but the final answer
    is a different quantity than the one asked: stopped at an intermediate step, gave the increase instead of the
    final amount, per-item instead of total, the wrong unit, etc.
  - `SETUP`: the first wrong value is not an arithmetic slip. The written calculation is computed correctly, but it is
    the wrong thing to compute: a wrong number taken from the problem or from an earlier line, a wrong quantity,
    relation, operation or formula, a missing or extra component, a misread condition.
  - `OTHER`: none of the above, e.g. no wrong value can be located (only an answer, no derivation); the right final
    value was computed but a different number that is not a valid quantity is given as the answer; the gold answer
    looks wrong; the solution is cut off before an answer.
- **num**: the line `L#` where the first wrong value is written, in an equation or in plain words. `TARGET`: the line
  where the wrong quantity is first given as the final answer. `OTHER`: the most relevant line, or `-` if none.
- **thought**: the line `L#` where the misunderstanding first shows in words, before any wrong number, e.g.
  "Her parents: 2 adults" (the parents are seniors), "1/3 of the cake was eaten" (the problem says 1/3 is left).
  If there is no earlier sign, the same as `num`. Always `thought` <= `num`. `CALC`: normally the same as `num`.
  `OTHER` with `num = -`: `-`.
- **conf**: `h` clear; `m` another choice is plausible: then write `alt=CLASS` and/or `alt=L#` in the reason.
- **reason**: at most 15 English words: what is wrong at `num`.

## Rules

1. Judge the values, not the style: unit words, rounding to the shown precision and restating an earlier result are
   fine.
2. A different but valid method is fine, and so are its intermediate values. A correct but unnecessary side
   computation is fine.
3. A value that is wrong only because an earlier value was wrong is not the first error: pick the earliest.
4. Copying a number wrongly (from the problem or from an earlier line) is `SETUP`, not `CALC`: `CALC` means the
   written arithmetic itself is wrong.
5. If the final answer happens to equal a number in the reference solution but an earlier value is wrong, it is not
   `TARGET`: classify by the first wrong value.
6. A line can contain several numbers: give the line of the first wrong one.
7. Judge the math yourself. Do not try to guess what an automatic checker would pick.
