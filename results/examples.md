# Worked examples (test set)

## D1000
```
E1 (Day 1): feverish
E2 (Day 2): temp 38.5, running on empty
E3 (Day 3): ate pasta, lost my keys again
E4 (Day 4): exhausted, temp 38.5, walked the dog
E5 (Day 5): temp 38.5
E6 (Day 6): hot and shivery, knees aching
E7 (Day 7): same as yesterday
E8 (Day 8): tired all day
E9 (Day 9): work was busy
E10 (Day 10): good day
E11 (Day 11): feverish, running on empty, achy joints, long shift
E12 (Day 12): walked the dog, bit of knees aching, temp 38.5
E13 (Day 13): saw friends, no vomiting, achy joints
E14 (Day 14): bit of achy joints, watched a film, tired of waiting for my results
```
**Ground truth:** fatigue, fever, joint pain | escalate: True
**LLM only:** The patient experienced multiple symptoms including fever, fatigue, and joint pain over several days. | escalate: False
**Agent v1:** The patient has been asymptomatic over the past 14 days. There have been no reported red flags or concerning symptoms. | escalate: False
**TrustTrace v2:** The patient experienced fever on days 1 through 14, with an improving trend. They also had abdominal pain from day 12 onwards, fatigue from days 2 to 14, joint pain worsening from day 11 to day 14, and poor sleep persisting from day 11. The fever is noted as a red flag. | escalate: True | rejected claims: {'negated': 1}
**TrustTrace v3:** The patient has experienced fatigue over the past 14 days, with symptoms stabilizing on day 14. Fever improved from day 1 to day 12 but worsened again on day 13 before returning to stable levels by day 14. Joint pain is worsening, most notably starting on day 6 and continuing through day 14. | escalate: True | judge rejections: {'judge said no (kw)': 3, 'judge said no (llm)': 5}

## D1001
```
E1 (Day 1): felt like throwing up, had red wine last night, tired all day
E2 (Day 2): fine
E3 (Day 3): achy joints, lost my keys again, felt queasy, long shift
E4 (Day 4): exhausted
E5 (Day 5): wiped out, work was busy
E6 (Day 7): exhausted, nauseous, achy joints
E7 (Day 6): long shift, nauseous, mild wrists sore
E8 (Day 9): running on empty, walked the dog
E9 (Day 10): knackered
E10 (Day 11): knackered
E11 (Day 12): wrists sore, work was busy, running on empty
E12 (Day 13): walked the dog, tired all day
```
**Ground truth:** fatigue, joint pain, nausea | escalate: False
**LLM only:** The patient experienced multiple symptoms including fatigue, nausea, vomiting, diarrhea, abdominal pain, and blood in stool starting from Day 7 onwards. | escalate: False
**Agent v1:** The patient has experienced fatigue for the past 14 days, with no significant change in intensity. Nausea also persisted for seven of those days without any notable improvement. | escalate: False
**TrustTrace v2:** The patient has experienced fatigue persisting from day 1 to day 13 with a stable trend. Joint pain is improving since day 3, while nausea has been gradually decreasing over the past few days. There are no red flags noted in the symptom progression. | escalate: False | rejected claims: none
**TrustTrace v3:** The patient has experienced fatigue persisting from day 1 to day 13 with a stable trend, joint pain improving since day 3, and nausea also showing improvement starting from day 1. No red flags have been identified. | escalate: False | judge rejections: {'judge said no (kw)': 6, 'judge said no (llm)': 4}

## D1002
```
E1 (Day 1): saw friends, bit of achy joints, temp 38.5, bathroom 6 times
E2 (Day 3): temp 38.5, mild wrists sore
E3 (Day 6): work was busy
E4 (Day 8): no runs today, hot and shivery, wrists sore
E5 (Day 9): no nausea, lost my keys again, wrists sore
E6 (Day 5): on the loo all morning, bit of wrists sore, walked the dog
E7 (Day 10): really bad wrists sore
E8 (Day 12): diarrhea, work is a pain in the neck, ate pasta
E9 (Day 13): temp 38.5, walked the dog, loose stools
E10 (Day 14): good day
```
**Ground truth:** diarrhoea, fever, joint pain | escalate: True
**LLM only:** The patient experienced abdominal pain, diarrhea, joint pain, and fatigue starting from Day 1. The symptoms worsened on Days 8-14 with increased frequency of bowel movements and worsening joint pain. | escalate: False
**Agent v1:** The patient experienced abdominal pain that was stable over the past 14 days. There were no red flags noted. | escalate: False
**TrustTrace v2:** The patient has experienced worsening diarrhoea over the past 14 days, with fever persisting from day 1 to day 13. Abdominal pain is stable, joint pain also remains unchanged, nausea persists on days 12 and 13, and poor sleep trends towards stability. The presence of fever is noted as a red flag. | escalate: True | rejected claims: {'negated': 1}
**TrustTrace v3:** The patient has experienced stable diarrhea over the past 14 days, with fever also remaining stable during that period. There have been no new red flags noted such as joint pain. | escalate: True | judge rejections: {'judge said no (llm)': 6, 'judge said no (kw)': 3}
