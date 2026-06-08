# 20 Newsgroups doc #6683 — figure-source data

- LLM: ERNIE-4.5-0.3B-PT
- DST: top-k=20, T=3.0
- BoW vocab size: 2000
- Label: 10  (rec.sport.hockey)
- Top-15 IoU(BoW, DST): 0.00

## Document

```
Obviously some reporter for the Ottawa Sun got taken by an April
Fools joke...probably started by someone with the Nordiques or the
Bruins.  

Like for example...who is going to reimburse the Flyers for the
$15 million they paid to the Nordiques...like the Senators are
going to get Lindros and $15 million.  The Flyers sent the
equivalent of 6 or 7 players (when you include the draft choices)
to Quebec, and they are going to get only four back.

Some reporter was had real badly and someone must be having a
real good laugh seeing as how the so much of the sports media
has chosen to publicize this utter nonsense.


Can you think...it cannot possibly be true...no need for the "if"!


I can't believe that anyone would consider giving such crap even
the remotest consideration.

Topic:
```

## BoW target — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `going` | 0.0909 |
| 2 | `million` | 0.0606 |
| 3 | `real` | 0.0606 |
| 4 | `paid` | 0.0303 |
| 5 | `good` | 0.0303 |
| 6 | `true` | 0.0303 |
| 7 | `having` | 0.0303 |
| 8 | `need` | 0.0303 |
| 9 | `draft` | 0.0303 |
| 10 | `include` | 0.0303 |
| 11 | `chosen` | 0.0303 |
| 12 | `like` | 0.0303 |
| 13 | `think` | 0.0303 |
| 14 | `consider` | 0.0303 |
| 15 | `believe` | 0.0303 |

## DST target (Ours) — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `sports` | 0.1621 |
| 2 | `hockey` | 0.0867 |
| 3 | `public` | 0.0676 |
| 4 | `report` | 0.0648 |
| 5 | `media` | 0.0474 |
| 6 | `news` | 0.0455 |
| 7 | `discussion` | 0.0455 |
| 8 | `newsgroups` | 0.0455 |
| 9 | `newsgroup` | 0.0455 |
| 10 | `justice` | 0.0427 |
| 11 | `article` | 0.0410 |
| 12 | `debate` | 0.0362 |
| 13 | `general` | 0.0347 |
| 14 | `social` | 0.0347 |
| 15 | `ice` | 0.0347 |
