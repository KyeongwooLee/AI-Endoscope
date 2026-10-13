# Follow-up pilot examples
Excluded from training and all evaluation; implementation/feasibility evidence only.

## gsm8k_train_5375
Jason has three times as many toys as John. If John has 6 more toys than Rachel and Rachel has 1 toy, how many toys does Jason have?

Official gold: 21; A error label: 0

```text
Reasoning:
Let's denote the number of toys Rachel has as R. According to the information given:

R = 1

John has 6 more toys than Rachel, so we can write:

J = R + 6

Substituting the value of R:

J = 1 + 6
J = 7

Jason has three times as many toys as John, so we can write:

Z = 3 * J

Substituting the value of J:

Z = 3 * 7
Z = 21

Final answer: 21
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 21 | 1 |
| Recalculate | False | None | None |
| Rebuild | True | 21 | 1 |
| Review | True | 21 | 1 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "21",
    "span": [
      349,
      351
    ],
    "status": "final_marker"
  },
  "token_indices": [
    116,
    117
  ],
  "token_ids": [
    17,
    16
  ],
  "decoded_tokens": [
    "2",
    "1"
  ],
  "mean_nll": 0.00012874603271484375,
  "max_nll": 0.00022125244140625,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 171,
    "token_id": 16,
    "decoded": "1",
    "offset": [
      599,
      600
    ]
  },
  "boundary": {
    "index": 172,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      600,
      602
    ]
  },
  "preverdict": {
    "index": 193,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      707,
      708
    ]
  }
}
```

## gsm8k_train_2326
Amanda’s garden contains 20 flowers and Peter’s garden contains three times as many flowers as Amanda's. If Peter gave 15 flowers to his brother, how many flowers are left in his garden?

Official gold: 45; A error label: 0

```text
Reasoning:
First, calculate the total number of flowers in Peter's garden before giving any away.
Peter has 3 * 20 = 60 flowers initially.
Then subtract the 15 flowers he gave to his brother.
So, 60 - 15 = 45 flowers remain in Peter's garden.

Final answer: 45.
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 45 | 1 |
| Recalculate | False | None | None |
| Rebuild | True | 45 | 1 |
| Review | True | 45 | 1 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "45",
    "span": [
      258,
      260
    ],
    "status": "final_marker"
  },
  "token_indices": [
    72,
    73
  ],
  "token_ids": [
    19,
    20
  ],
  "decoded_tokens": [
    "4",
    "5"
  ],
  "mean_nll": 0.0001163482666015625,
  "max_nll": 0.0002231597900390625,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 137,
    "token_id": 20,
    "decoded": "5",
    "offset": [
      562,
      563
    ]
  },
  "boundary": {
    "index": 139,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      564,
      566
    ]
  },
  "preverdict": {
    "index": 160,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      671,
      672
    ]
  }
}
```

## gsm8k_train_5027
Collin learned that he could earn $0.25 per aluminum can that he brought into the recycling center.  He found 12 cans at home and three times as many at his grandparents' house.   His neighbor gave him 46.  His dad brought home 250 cans from the office.  His dad told him he could keep the money from the cans he brought home from the office if he promised to put half of his entire amount collected into savings.  How much money would Collin have to put into savings?

Official gold: 43; A error label: 1

```text
Reasoning:
First, calculate the total number of cans Collin found at home and at his grandparents' house:
- Cans at home: 12
- Cans at grandparents' house: 3 * 12 = 36

Total cans from home and grandparents' house: 12 + 36 = 48

Next, add the cans given by his neighbor:
Cans from neighbor: 46

Now, add the cans brought home by his dad:
Cans from dad's office: 250

Total cans collected: 48 (home) + 36 (grandparents) + 46 (neighbor) + 250 (dad's office) = 400

Since Collin earns $0.25 per can, calculate the total earnings:
Earnings: 400 * $0.25 = $100

According to his dad's condition, Collin needs to save half of his total earnings:
Amount to save: $100 / 2 = $50

Final answer: 50.0
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 5E+1 | 0 |
| Recalculate | True | 5E+1 | 0 |
| Rebuild | True | 2 | 0 |
| Review | True | 5E+1 | 0 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "5E+1",
    "span": [
      686,
      690
    ],
    "status": "final_marker"
  },
  "token_indices": [
    230,
    231,
    232,
    233
  ],
  "token_ids": [
    20,
    15,
    13,
    15
  ],
  "decoded_tokens": [
    "5",
    "0",
    ".",
    "0"
  ],
  "mean_nll": 0.10063648223876953,
  "max_nll": 0.2854480743408203,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 366,
    "token_id": 15,
    "decoded": "0",
    "offset": [
      1274,
      1275
    ]
  },
  "boundary": {
    "index": 367,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      1275,
      1277
    ]
  },
  "preverdict": {
    "index": 388,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      1382,
      1383
    ]
  }
}
```

## gsm8k_train_5241
Chloe bought chocolate-dipped strawberries at $50 a dozen. She then sold them for $30 for half a dozen during the Mother's Day celebration. How much is Chloe's profit if she sold 50 dozens?

Official gold: 5E+2; A error label: 1

```text
To calculate Chloe's profit:

1. Calculate total cost of buying 50 dozens of strawberries.
   Cost per dozen = $50
   Total cost = 50 * $50 = $2500

2. Calculate total revenue from selling 50 dozens of strawberries.
   Selling price per half dozen = $30
   Number of half dozens in 50 dozens = 50 / 2 = 25
   Total revenue = 25 * $30 = $750

3. Calculate profit by subtracting total cost from total revenue.
   Profit = Total Revenue - Total Cost
         = $750 - $2500
         = -$1750

Therefore, Chloe's loss is $1750.

Final answer: -1750.
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | -1.75E+3 | 0 |
| Recalculate | False | None | None |
| Rebuild | True | 5E+2 | 1 |
| Review | True | -1.75E+3 | 0 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "-1.75E+3",
    "span": [
      539,
      544
    ],
    "status": "final_marker"
  },
  "token_indices": [
    169,
    170,
    171,
    172,
    173
  ],
  "token_ids": [
    481,
    16,
    22,
    20,
    15
  ],
  "decoded_tokens": [
    " -",
    "1",
    "7",
    "5",
    "0"
  ],
  "mean_nll": 0.037285995483398435,
  "max_nll": 0.17865371704101562,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 241,
    "token_id": 15,
    "decoded": "0",
    "offset": [
      849,
      850
    ]
  },
  "boundary": {
    "index": 243,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      851,
      853
    ]
  },
  "preverdict": {
    "index": 264,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      958,
      959
    ]
  }
}
```

## gsm8k_train_2121
Nancy wanted to make peanut butter cookies for a family gathering, but her cousin is allergic to peanuts. She decided to make almond butter cookies instead. A jar of almond butter costs three times the amount that a jar of peanut butter does. It takes half a jar to make a batch of cookies. A jar of peanut butter costs $3. How many dollars more does it cost per batch to make almond butter cookies instead of peanut butter cookies?

Official gold: 3; A error label: 1

```text
To solve this problem, let's break it down step by step:

1. **Cost of Peanut Butter Jars**: 
   - Given: 1 jar of peanut butter costs $3.
   
2. **Cost of Almond Butter Jars**:
   - Since almond butter costs three times as much as peanut butter,
     \[
     \text{Cost of almond butter} = 3 \times \$3 = \$9
     \]

3. **Amount of Almond Butter Needed Per Batch**:
   - It takes half a jar to make a batch of cookies.
     \[
     \text{Half a jar} = \frac{1}{2} \text{ jar}
     \]
     
4. **Cost Calculation**:
   - Cost of almond butter per batch:
     \[
     \text{Cost of almond butter per batch} = \$9
     \]
   - Cost of peanut butter per batch (half a jar):
     \[
     \text{Cost of peanut butter per batch} = \frac{1}{2} \times \$3 = \$1.50
     \]

5. **Difference in Cost**:
   - To find how much more it costs to make almond butter cookies compared to peanut butter cookies per batch,
     \[
     \text{Difference} = \$9 - \$1.50 = \$7.50
     \]

Therefore, it costs Nancy $\boxed{7.50}$ dollars more per batch to make almond butter cookies instead of peanut butter cookies.
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 7.5 | 0 |
| Recalculate | False | None | None |
| Rebuild | True | 3 | 1 |
| Review | True | 7.5 | 0 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "7.5",
    "span": [
      1003,
      1007
    ],
    "status": "boxed"
  },
  "token_indices": [
    299,
    300,
    301,
    302
  ],
  "token_ids": [
    22,
    13,
    20,
    15
  ],
  "decoded_tokens": [
    "7",
    ".",
    "5",
    "0"
  ],
  "mean_nll": 0.011957645416259766,
  "max_nll": 0.04738616943359375,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 412,
    "token_id": 15,
    "decoded": "0",
    "offset": [
      1555,
      1556
    ]
  },
  "boundary": {
    "index": 429,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      1645,
      1647
    ]
  },
  "preverdict": {
    "index": 450,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      1752,
      1753
    ]
  }
}
```

## gsm8k_train_45
It's Ava's birthday party. Her parents bought a unicorn piñata for $13 and filled it with all of her favorite treats. They bought 4 bags of Reese's for $9 per bag, 3 bags of Snickers for $5 per bag, and 5 bags of Skittles for $7 per bag. How much did the unicorn piñata and the treats cost altogether?

Official gold: 99; A error label: 0

```text
Reasoning:
First, calculate the total cost of the Reese's: 4 bags * $9/bag = $36.
Next, calculate the total cost of the Snickers: 3 bags * $5/bag = $15.
Then, calculate the total cost of the Skittles: 5 bags * $7/bag = $35.
Now, add up all the costs to find the total amount spent on the piñata and treats: $13 (piñata) + $36 (Reese's) + $15 (Snickers) + $35 (Skittles) = $99.

Final answer: 99.0
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 99 | 1 |
| Recalculate | False | None | None |
| Rebuild | True | 9E+1 | 0 |
| Review | True | 99 | 1 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "99",
    "span": [
      392,
      396
    ],
    "status": "final_marker"
  },
  "token_indices": [
    141,
    142,
    143,
    144
  ],
  "token_ids": [
    24,
    24,
    13,
    15
  ],
  "decoded_tokens": [
    "9",
    "9",
    ".",
    "0"
  ],
  "mean_nll": 0.22796344757080078,
  "max_nll": 0.6879329681396484,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 248,
    "token_id": 15,
    "decoded": "0",
    "offset": [
      813,
      814
    ]
  },
  "boundary": {
    "index": 249,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      814,
      816
    ]
  },
  "preverdict": {
    "index": 270,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      921,
      922
    ]
  }
}
```

## gsm8k_train_7360
Apollo pulls the sun across the sky every night. Once a month, his fiery chariot’s wheels need to be replaced. He trades golden apples to Hephaestus the blacksmith to get Hephaestus to make him new wheels. Hephaestus raised his rates halfway through the year and now demands twice as many golden apples as before. He charged three golden apples for the first six months. How many golden apples does Apollo have to pay for the entire year of chariot wheels?

Official gold: 54; A error label: 0

```text
Reasoning:
For the first 6 months, Apollo pays 3 golden apples per wheel.
For the remaining 6 months, he pays 2 * 3 = 6 golden apples per wheel.
Total golden apples paid in the first half of the year: 3 golden apples/wheel * 6 wheels = 18 golden apples.
Total golden apples paid in the second half of the year: 6 golden apples/wheel * 6 wheels = 36 golden apples.
Total golden apples paid for the entire year: 18 + 36 = 54 golden apples.

Final answer: 54
```

| Action | Available | Extracted answer | Success |
|---|---|---|---|
| Keep | True | 54 | 1 |
| Recalculate | True | 54 | 1 |
| Rebuild | True | 108 | 0 |
| Review | True | 54 | 1 |

Numerical-token measurement:
```json
{
  "parsed": {
    "answer": "54",
    "span": [
      453,
      455
    ],
    "status": "final_marker"
  },
  "token_indices": [
    126,
    127
  ],
  "token_ids": [
    20,
    19
  ],
  "decoded_tokens": [
    "5",
    "4"
  ],
  "mean_nll": 8.869171142578125e-05,
  "max_nll": 0.0001678466796875,
  "available": true
}
```
Pre-verdict positions:
```json
{
  "numeric": {
    "index": 248,
    "token_id": 19,
    "decoded": "4",
    "offset": [
      1027,
      1028
    ]
  },
  "boundary": {
    "index": 249,
    "token_id": 271,
    "decoded": "\n\n",
    "offset": [
      1028,
      1030
    ]
  },
  "preverdict": {
    "index": 270,
    "token_id": 198,
    "decoded": "\n",
    "offset": [
      1135,
      1136
    ]
  }
}
```
