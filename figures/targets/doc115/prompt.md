# 20 Newsgroups doc #115 — figure-source data

- LLM: ERNIE-4.5-0.3B-PT
- DST: top-k=20, T=3.0
- BoW vocab size: 2000
- Label: 0
- Top-15 IoU(BoW, DST): 0.00

## Document

```
Issued by Khomeini it shouldn't be relevant to anyone. But issued
by an honest and learned scholar of Islam it would be relevant to
any muslim as it would be contrary to Islamic law which all muslims
are required to respect.


Anyone sufficiently well versed in Islamic law and capable of reasoning,
if you are talking about a weak sense of "excuse." It depends on what 
sense of "excuse" you have in mind.



Only someone who thinks my opinion is important, obviously.
Obviously you don't care, nor do I care that you don't care.
```

## BoW target — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `care` | 0.1154 |
| 2 | `excuse` | 0.0769 |
| 3 | `sense` | 0.0769 |
| 4 | `law` | 0.0769 |
| 5 | `relevant` | 0.0769 |
| 6 | `reasoning` | 0.0385 |
| 7 | `thinks` | 0.0385 |
| 8 | `respect` | 0.0385 |
| 9 | `depends` | 0.0385 |
| 10 | `mind` | 0.0385 |
| 11 | `contrary` | 0.0385 |
| 12 | `required` | 0.0385 |
| 13 | `weak` | 0.0385 |
| 14 | `capable` | 0.0385 |
| 15 | `important` | 0.0385 |

## DST target (Ours) — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `discussion` | 0.0940 |
| 2 | `religious` | 0.0717 |
| 3 | `debate` | 0.0620 |
| 4 | `argument` | 0.0607 |
| 5 | `discussing` | 0.0570 |
| 6 | `opinion` | 0.0558 |
| 7 | `understanding` | 0.0514 |
| 8 | `religion` | 0.0503 |
| 9 | `political` | 0.0483 |
| 10 | `legal` | 0.0463 |
| 11 | `moral` | 0.0435 |
| 12 | `analysis` | 0.0426 |
| 13 | `issue` | 0.0426 |
| 14 | `question` | 0.0417 |
| 15 | `morality` | 0.0408 |
