# YAICON 11.21. 공유자료 - 정성훈

# 이전까지 시도해온 방법들의 문제점

## 1. Clustering이 잘 안됨

다시 생각해보니 해당 방법자체가 잘못되었다는 생각이 들었습니다

- 이전에 다현님도 말씀하셨던 것 처럼 model이 틀린 문제에서는 풀이에서 여러 오류 유형이 동시에 나타나는 경우가 많음
- 뿐만 아니라 모델이 그냥 풀이를 작성하지 않은 경우에는 다른 오류 유형(e.g. 관계 유형 틀림, 계산식 틀림)도 부여해야하는지 애매한 부분이 있음

⇒ 그러나 clustering은 하나의 오답에 하나의 오류 유형만 나타난다고 전제하기에(하나의 오답에 하나의 오류 유형 class만 부여하기에) 틀린 접근방법 아닐까?

근데 이전에 clustering을 해봤을 때 여러 cluster중 확실한 오류 유형을 지니고 있는 case도 있었습니다

- 모델이 그냥 풀이를 짧게 작성하거나, 풀이를 작성하지 않고 답변만 ouptut한 경우는 하나의 군집으로 잘 모였었음

⇒ 풀이 미작성이라는(reasoning을 output하지 않음) 경우는 다른 오류 유형과 구분되는 확실한? 오류 유형 아닐까?

## 2. 만약 우리가 특정 유형의 오류를 탐지했다하더라도 어떠한 방식으로 Correction을 할 수 있는 방법이 모호함

- 계산식을 잘 세웠는데 계산과정에서 오류가 발생한 오류 유형(산술 실행 오류)는 게산기를 사용해서 고칠 수 있다

⇒ 그럼 다른 오류 유형은?

좀 찾아보니깐 최신 모델에서는 이미 model이 token 생성의 방식으로 계산식을 세우면 해당 계산 식은 계산기를 tool로 불러와서 사용하는 경우도 있는 것 같더라고요

뿐만 아니라 저희가 오류 유형마다 해당 오류를 수정할 수 있는 명확한 방법을 찾는다 하더라도 저희가 임의로 나눈 모델 오류 유형에 따른 적합한 방법을 Heuristic하게 찾는 건 현실적인 방법이 아니라는 생각이 들었습니다

⇒RL 방식을 사용해서 reward model로 모델을 판별하고 모델이 정상 trajectory에서 벗어났을때(잘못된 방식으로 풀이를 생성/ 문제를 풀고 있을 때) 해당 reward model을 기준으로 높은 reward를 받게하는 gradient를 계산해서 해당 gradient를 바탕으로 inference 중간에 correction할 순 없나?

참고논문 : SCoRe(**S**elf-**Co**rrective **Re**inforcement Learning)[https://arxiv.org/abs/2409.12917](https://arxiv.org/abs/2409.12917),

InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning [https://arxiv.org/abs/2601.14209](https://arxiv.org/abs/2601.14209)

## 3. 우리가 임의로 설정한 오류 유형을 기준으로 LLM Judge를 이용한 결과물을 인간검수 없이 그대로 사용해도될까?

labeling을 하면서 해당 사항이 궁금해져서 직접 한번 비교해봤습니다

비교 대조군은



![image.png](attachment:e1107589-407d-4e89-a825-34b025bf15d9:image.png)

- GPT-6.2-Sol
- GPT-6.1-Terra
- GPT-6-Luna

이전 GSM8K dataset이 Qwen 1.5B model에게 이미 너무 쉬운 문제인 것 같아서(4B model을 쓰면 오답이 거의 안나옴) 조금 더 어려운 benchmark인 MATH set 을 Qwen 1.5B 모델로 풀린 다음(정답률 50% 정도 나옴) 나온 동일한 오답 500개를 세 모델에게 labeling시켜보고 결과를 비교해봤습니다

### 결과



![image.png](attachment:2aa655ef-7bc7-4b2c-afec-d4fe7c2fef51:image.png)

오류 유형을 labeling할 때 이전과 달리 여러 오류 유형을 중복으로 labeling하도로 했습니다

- API로 불러서 오답 유형 labeling시켰을 때 드는 비용(크레딧 남는게 조금 있어서 그걸로 좀 돌려봤습니다)

  
  ![image.png](attachment:9c1158f4-f68b-456c-be4f-616706bdea5b:image.png)

⇒ 해당 모델들이 다들 나름 최신 모델들인데도 불구하고 오류 유형을 분류하는데 차이가 생각보다 크더라고요 그래서 일단 6.2 Sol 이용해서 최대한 labeling해보고 있긴한데 이게 토큰 사용량이 상당해서ㅠㅠ(GSM8K 문제와 달리 MATH 문제를 풀리면 문제 난이도가 상대적으로 더 어렵다보니깐 Qwen이 생성한 output의 길이가 길어지는 경우가 많아서 그런 것 같아요) 일단 최대한 많이 돌려보겠습니다

# So…

[해당 논문](https://app.notion.com/p/YAICON-11-21-41c78a7dac468011a437c84e40c506f7?pvs=21)을 참고하면서 조금 접근 방법을 틀어서 해보려고 합니다

1. **Reward model을 학습시키되, 오류 유형별로 따로따로 학습된 Reward model(e.g. 관계 유형을 잘 지키고 있는지 판별하는 reward model, 계산 식을 잘 수립하고 있는지 판별하는 reward model)을 사용함**

  이전에 언급했듯, 오류 유형별로 해당 오류를 correction할 수 있는 방법이 다르지 않을까라고 생각했습니다

  예상되는 해당 구조의 장점은 아래와 같습니다
  - Reward model을 잘 학습시키기 좀 더 쉽지 않을까?
  
    단순히 풀이를 잘 작성하고 있음/풀이를 잘 작성하지 못하고 있음 이라는 큰 task를 다루는 것 보다는 task를 세분화 해서 특정 오류 유형이 나타남/나타나지 않음 을 판별하는 것이 reward model 입장에서 조금 더 학습하기 쉬운(handle하기 쉬운?) task 일것이라 생각합니다
  - 오류 유형마다 적절히 correction하는 방법이 조금씩 다르지 않을까?
  
    기존 논문에서 제시한 방법처럼 reward model을 통해 correct trajectory로 향하는 gradient를 추출해서 사용하되, 특정 오류 유형에 맞는 올바른 gradient를 찾아 조금 더 detail하게 수정하기에 조금 더 좋은 결과가 나올 수 있지 않을까 라는 생각이 들었습니다
2. **논문처럼 reward model의 input으로 SAE를 통해 추출한 latent vector를 주는 것이 아닌 그냥 model의 hidden state를 그대로 reward model의 input으로 사용함**

  논문에서는 아마 계산량을 좀 줄이기 위해(가벼운 reward model을 사용하기 위해?) SAE로 latent vector를 뽑고 해당 vector를 바탕으로 reward 계산을 하고, correction을 할때는 decoder를 통해 latent vector를 바탕으로 다시 hidden state를 복구해서 사용하는 구조였습니다  
  ⇒ gradient를 계산하는 입장에서 decoder로 hidden state를 복구하여 사용하면 정보 손실이 크게 일어나지 않을까 싶어서 그냥 hidden state에다가 바로 연결하려고 합니다

  이렇게 되면 계산량이 너무 커질수도 있을 것 같은데(아직 제대로 측정안해봐서 모르겠긴한데 아마 기존 논문의 방법보다는 더 많을 것 같긴합니다…) 매 time step마다 reward를 계산하는 것이 아닌 token 4개를 생성할 때 마다 reward를 계산하고, 만약 reward가 threshold 이하가 된다면 이전 4 time step(4개의 token 생성 전)으로 돌아가서 gradient를 바탕으로 correction을 한다던지…하는 방법으로 생각하고 있습니다
3. **LLM Judge로 labeling을 하되 현실적으로 타협해서 6.2 Sol로**

  LLM Judge로 어떤 model을 사용하느냐에 따라 결과가 생각보다 확확 달라져서 최대한 좋은 모델로 판별하려 합니다

  그렇다고 Astra로 오답 전부 labeling하면 몇백만원이 들거같아서.. 일단 제가 GPT Plus plan가진걸로 최대한 뽑아보고 안되면 크레딧 좀 남는걸로도 마저 해보겠습니다

  아님 preference learning시켜서 특정 모델 fine-tuning한 다음에 해당 model GPU로 돌려서 판별하는 방법도 있긴한데, 시간도 좀 많이들고 앵간한 fine-tuning한 모델보다 그냥 frontier model돌리는게 더 나을 것 같다는 생각도 드네요
4. **Reward model을 학습할때 풀이를 작성하지 않은 오류유형은 masking해서 학습**

  특정 오류 유형의 reward model을 학습할때 특정 오류유형이 나타남/특정 오류 유형이 나타나지 않음 의 이진 분류 후 training하려고 하는데 이때 풀이를 작성하지 않은 오류 유형은 항상 정답으로 간주되어 training되기에 해당 오류 유형은 reward model training에 사용하지 않으려고 합니다

  근데 지금 오답을 좀 비교해봤을때 풀이를 안적는 경우가 없었긴합니다ㅠ

# 현황(11.21. 4PM 기준)

## 오류 유형 labeling

일단 MATH data inference 돌려서 정오답 데이터 각 15000개 정도씩 뽑아놓고 오류유형 labeling했는데 (GPT-6.2-Sol로) 이전에 사용하던 오류 유형들 중 관계·조건·목표 구성 `modeling_relation` , 중간 결과·조건 추적 `reasoning_state` , 계산 실행 `arithmetic_execution` 이 주로 나타나고 나머지 오류 유형은 잘 나타나지 않아서 우선 세 유형으로만 오류 유형 labeling했습니다


| 오류 유형                           | 개수     |
| ------------------------------- | ------ |
| 관계·조건·목표 구성 `modeling_relation` | 9,379  |
| 중간 결과·조건 추적 `reasoning_state`   | 11,306 |
| 계산 실행 `arithmetic_execution`    | 5,686  |


⇒ 이러한 오류 유형별로 reward model 3개를 학습시켰습니다

## Reward model 학습 결과


| Reward head            | Holdout AUROC | 개입 임계값 |
| ---------------------- | ------------- | ------ |
| `modeling_relation`    | 0.909         | 0.9513 |
| `reasoning_state`      | 0.884         | 0.9466 |
| `arithmetic_execution` | 0.814         | 0.9468 |


전체 정답 data 12500개 정도랑 오답 data 12500개 8:1:1 비율로 train/validation/test set으로 나눠서 학습하고 나온 결과입니다

제가 고안한 architecutre에서는 token을 생성할 때 reward model 값을 계산하면서 reward가 해당 임계값 아래로 떨어지면 해당 reward model을 바탕으로 gradient correction하는 방식이에용

## 실제 실험 결과

해당 구조에서 hyperparameter로 둘 수 있는 값들은

- reward model을 통해 correction을 진행할 때(hidden state값을 바꿀 때) reward model을 통해 계산한 gradient를 얼마나 반영할지 선택하는 개입 강도 $\alpha$
- reward model을 몇 token마다 계산할지 고르는 time step $t$
- 그리고 reward model을 통해 correction을 할때 해당 개입 시점 $t'$에서 이전 time step $t'-t$만큼 돌아가서 생성할지(롤백), 아니면 $t'$에서부터 마저 다시 생성할지 선택하는 롤백여부 $r$

정도가 있습니다

### 개입 강도 $\alpha$실험


| 개입 강도 α  | 정답률       | 롤백 후 토큰이 바뀐 구간 |
| -------- | --------- | -------------- |
| 무개입      | 57.0%     | —              |
| 0.02     | 56.8%     | 23             |
| 0.05     | 56.2%     | 54             |
| 0.10     | 56.4%     | 91             |
| 0.20     | 57.4%     | 205            |
| **0.40** | **58.6%** | 389            |
| 0.60     | 57.0%     | 573            |
| 0.80     | 56.6%     | 837            |
| 1.00     | 55.2%     | 1,093          |


- time step $t = 4$
- 롤백 여부 $r = true$
- Greedy 방식으로 token 생성($\neq$ sampling방식, input이 동일하다면 output도 항상 동일함)

처음에는 $\alpha$값을 엄청 작게(0.02)정도로 정하고 inference돌려봤을 때 gradient correction을 하더라도 롤백 이전에 생성한 토큰이랑 동일하게 생성하는 경우가 대부분이더라고요(gradient correction을 1000번 정도 하는데 실제 생성 토큰이 롤백 이전과 달라지는 경우는 10번 정도 내외)

그래서 개입 강도를 좀 확확 키워보면서 실험해봤는데 $\alpha = 0.4$일 때 성능이 가장 괜찮게 나왔던 것 같습니다

### time step $t$, 롤백 여부 $r$ 실험


| Reward 계산 간격 | 롤백 없이 즉시 개입     | 롤백 후 재생성            |
| ------------ | --------------- | ------------------- |
| 무개입          | 50.2% (251/500) | —                   |
| 1토큰          | 50.8% (254/500) | 51.6% (258/500)     |
| 2토큰          | 51.2% (256/500) | **52.0% (260/500)** |
| 4토큰          | 49.8% (249/500) | 51.8% (259/500)     |


- 개입강도 $\alpha = 0.4$
- Sampling 방식으로 token 생성( $\neq$ Greedy방식)
  - temperature = 0.7
  - top-p = 0.95

Sampling 방식으로 토큰을 다양하게 생성하는거랑 롤백을 통해 토큰을 다양하게 생성하는거랑 원인을 분리하기 위해 sampling방식으로 돌렸습니다

그리고 롤백을 하는 게 나을지, time step은 얼마정도로 잡을지 실험 결과 비교해봤을때는 확실히 $r = true$로 두는 방식이 계산량이 늘어날지언정, 결과는 괜찮았던 것 같고, $r = true$ 일때는 $t$가 그렇게 큰 영향을 미치는 것 같진 않았습니다  
의외인 경우는 time step $t$가 작을수록 계산량이 늘어나더라도 결과가 제일 좋을줄 알았는데 마냥 그렇진 않은 것 같습니다

### 실제 correction 예시

$\alpha = 0.4, t = 2, r = true$, Greedy방식으로 출력한 결과

- 문제 원문

  &lt;aside&gt;  
  ❓

  A Senate committee has 5 Democrats, 5 Republicans, and 1 Independent. In how many ways can they sit around a circular table if all the members of each party all sit next to each other? (Two seatings are considered equivalent if one is a rotation of the other.)

  &lt;/aside&gt;

  ⇒ 실제 정답 : 28800
- 무개입 output

  &lt;aside&gt;  
  👶

  To solve the problem of seating 5 Democrats, 5 Republicans, and 1 Independent around a circular table such that all members of each party sit next to each other, we need to follow these steps:
  1. **Treat each party as a single "block":**
    - We have 3 blocks: one for Democrats, one for Republicans, and one for the Independent.
    - Since the blocks are indistinguishable within their own party, we can treat each block as a single unit.
  2. **Arrange the blocks around the table:**
    - In a circular arrangement, fixing one block to break the rotational symmetry, we have 2 remaining blocks to arrange.
    - The number of ways to arrange 2 blocks in a circle is \((2-1)! = 1!\).
  3. **Arrange the members within each block:**
    - Within each block, the members can be arranged among themselves.
    - The number of ways to arrange 5 Democrats is \(5!\).
    - The number of ways to arrange 5 Republicans is \(5!\).
    - The number of ways to arrange 1 Independent is \(1!\).
  4. **Combine the arrangements:**
    - The total number of ways to arrange the blocks and the members within each block is:  
    \[  
    1! \times 5! \times 5! \times 1!  
    \]
  5. **Calculate the factorials:**
    - \(1! = 1\)
    - \(5! = 120\)
  6. **Multiply the values:**  
  \[  
  1 \times 120 \times 120 \times 1 = 14400  
  \]

  Therefore, the number of ways to seat the members of the Senate committee around a circular table such that all members of each party sit next to each other is \(\boxed{14400}\).

  &lt;/aside&gt;
- Reward model 개입 output

  &lt;aside&gt;  
  🧞

  To solve the problem of seating 5 Democrats, 5 Republicans, and 1 Independent around a circular table such that all members of each party sit next to each other, we need to follow these steps:
  1. **Treat each party as a single "block":**
    - We have 3 blocks: one for Democrats, one for Republicans, and one for the Independent.
    - Since the blocks are indistinguishable within their own party, we can treat each block as a single unit.
  2. **Arrange the blocks around the table:**
    - In a circular arrangement, fixing one block to break the rotational symmetry, we (132번째 token에서 modeling_relation reward model 개입)can arrange the remaining(136번째 token에서 modeling_relation reward model 개입) 2 blocks in \(2!\) (2 factorial) ways.
    - Therefore, there are \(2!\) ways to arrange the 2 blocks.
  3. **Arrange the members within each block:**
    - Within each block, the members can be arranged among themselves. For the Democrats, there are \(5!\) ways to arrange them.
    - For the Republicans, there are \(5!\) ways to arrange them.
    - For the Independent, there(232번째 token에서 modeling_relation reward model 개입) are \(1!\) way to arrange them (since there is only one Independent).
  4. **Combine all the arrangements:**
    - The total number of ways to arrange the blocks and the members within each block is given by:  
    \[  
    2! \times 5! \times 5! \times 1!  
    \]
  5. **Calculate the factorials:**
    - \(2! = 2\)
    - \(5! = 120\)
  6. **Multiply the values:**  
  \[  
  2 \times 120 \times 120 \times 1 = (136번째 token에서 `reasoning_state` reward model, `arithmetic_execution` reward model 동시 개입)28800  
  \]

  Therefore, the total number of ways to seat the members around the circular table is \(\boxed{28800}\).

  &lt;/aside&gt;

  실제로 기존 풀이에서 틀린 부분이
  > we (132번째 token에서 modeling_relation reward model 개입)can arrange the remaining(136번째 token에서 modeling_relation reward model 개입) 2 blocks in \(2!\) (2 factorial) ways.

  여기였는데 해당 token 근처에서 correction이 들어가면서 해당 부분을 정답으로 잘 생성하더라고요
  - 그러나 기존에 오답이었던 \(2!\) 부분에서 correction이 들어가는 게 아닌 그보다 더 이전 token에서 correction이 들어가는 점이 좀 흥미로운 것 같습니다

# 이후 계획

지금 전체적으로 실험한 결과에서 진짜 획기적인 성능 개선은 잘 안보이지만 뭔가 가능성은 보이는 것 같아서 추가로 최대한 실험 더 돌려보겠습니다…

1. RL 방식의 reward model을 학습하는 것이 아닌 GRPO방식으로 tuning
2. hyperparameter tuning 조금 더 detail하게
3. Math dataset말고 다른 dataset으로도 돌려보기
  1. 다른 dataset으로 성능이 전이 가능한지 확인해보기
4. 계산량(FLOPs)을 줄일 수 있을 만한 방법 찾아보기

  지금 전체적으로 성능 개선은 1~2%대인데 추론 과정에서 FLOPs가 20% 정도씩 증가하는 상황이라.. 뭔가 최적화 할 수 있을 만한 방법을 고민해봐야할 것 같습니다

추가로 더 돌려볼만한 아이디어 주시면 감사하겠습니다!!