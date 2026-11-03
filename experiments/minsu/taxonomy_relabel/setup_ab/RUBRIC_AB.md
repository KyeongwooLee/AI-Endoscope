# Step 2 rubric: type of the first setup error (v1, 2026-11-01)

Each item is a WRONG solution of Qwen2.5-1.5B-Instruct to a GSM8K problem, shown next to a CORRECT solution of the same
problem by the same model and the GSM8K reference. Calculation slips and "answered an intermediate value" were already
removed by rules; what is left are setup errors. A rule found the **auto-detected first wrong step**: the first
equation whose result appears in no correct solution. Use the correct solutions to see what should have happened there.

## Fields

`id<TAB>loc<TAB>code<TAB>conf<TAB>reason`

- **loc**: `y` if the auto-detected step (or the sentence that sets it up) is the first real error; `n` if the first
  real error is clearly elsewhere. With `n`, label the real first error.
- **code**: one of the codes below.
- **conf**: `h` clear; `m` another code is plausible. With `m`, write `alt=XX` in the reason.
- **reason**: at most 15 English words.

## Codes

**A. Fact taken from the problem wrong: the VALUE, ENTITY or COUNT itself is wrong**

| code | when |
|---|---|
| CV | a number for a stated quantity is wrong: miscopied, or the number of a different entity/quantity is used (price of A used for B; 12 used where the problem's 60 belongs) |
| CO | things listed in the problem are miscounted: an entity, item, day or period is left out or counted twice (Mon–Thu as 5 days; forgets the queen) |
| HA | information that is not in the problem is added: invented fee, assumed starting value, extra condition |

**B. Facts taken right, but the relation, operation or plan is wrong**

| code | when |
|---|---|
| OP | wrong operation or direction between correct quantities: + / − swapped, × / ÷ swapped, "more/less than", "times / fraction of" reversed, profit vs loss sign |
| FC | wrong formula or relation model: percent base, rate × time, average, ratio, per-unit vs total relation, area vs perimeter |
| MS | a needed component or step is skipped while computing the asked quantity: a group, a cost, a return trip, a conversion; a relation word such as "2 more than" ignored entirely |
| TG | the final reported quantity is a different well-defined quantity than asked: a part instead of the total, increase instead of final, per-unit instead of total |

**C / D / other**

| code | when |
|---|---|
| PS | the model misuses its OWN earlier value: value before an update, a change applied twice, an earlier result copied wrong |
| UC | unit conversion missing or wrong, or different units mixed |
| UN | cannot tell, or no real error |

## Tie-break rules (apply in order)

1. Relation words (more/less than, times, half, of, per, each, percent off, left/remaining): ignoring or misapplying
   them is **B** (OP/FC/MS), never A.
2. **A** needs the value itself to be wrong compared with the problem: a number that is not the problem's number for
   that quantity, a number of a different entity, a wrong count of listed things, or invented information.
3. Right numbers combined wrongly → **B**.
4. The model's own earlier number used wrongly → **PS**.
5. An A and a B error in the same step: take the one that appears first in the text; if simultaneous, take A.
6. MS vs TG: the final reported quantity is not what the question asks → TG; a piece missing on the way → MS.
