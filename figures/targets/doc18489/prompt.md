# 20 Newsgroups doc #18489 — figure-source data

- LLM: ERNIE-4.5-0.3B-PT
- DST: top-k=20, T=3.0
- BoW vocab size: 2000
- Label: 14  (sci.space)
- Top-15 IoU(BoW, DST): 0.00

## Document

```
jfc> If gamma ray bursters are extragalactic, would absorption from the
jfc> galaxy be expected?  How transparent is the galactic core to gamma
jfc> rays?

and later...

JB> So, if the 1/r^2 law is incorrect (assume
JB> some unknown material [dark matter??] inhibits Gamma Ray propagation),
JB> could it be possible that we are actually seeing much less energetic
JB> events happening much closer to us?  The even distribution could
JB> be caused by the characteristic propagation distance of gamma rays 
JB> being shorter then 1/2 the thickness of the disk of the galaxy.


 0.

 Well, maybe not zero, but very little.  At the typical energies for 
 gamma rays, the Galaxy is effectively transparent. 

 Hans Bloemen had a review article in Ann. Rev. Astr. Astrophys. a few 
 years back in which he discusses this in more depth.

Topic:
```

## BoW target — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `typical` | 0.0370 |
| 2 | `zero` | 0.0370 |
| 3 | `ray` | 0.0370 |
| 4 | `article` | 0.0370 |
| 5 | `unknown` | 0.0370 |
| 6 | `maybe` | 0.0370 |
| 7 | `review` | 0.0370 |
| 8 | `years` | 0.0370 |
| 9 | `possible` | 0.0370 |
| 10 | `later` | 0.0370 |
| 11 | `happening` | 0.0370 |
| 12 | `assume` | 0.0370 |
| 13 | `little` | 0.0370 |
| 14 | `actually` | 0.0370 |
| 15 | `events` | 0.0370 |

## DST target (Ours) — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `dark` | 0.1086 |
| 2 | `understanding` | 0.0795 |
| 3 | `xterm` | 0.0569 |
| 4 | `discussion` | 0.0535 |
| 5 | `explanation` | 0.0502 |
| 6 | `energy` | 0.0502 |
| 7 | `light` | 0.0502 |
| 8 | `transmission` | 0.0492 |
| 9 | `explain` | 0.0462 |
| 10 | `question` | 0.0453 |
| 11 | `interpretation` | 0.0453 |
| 12 | `universe` | 0.0434 |
| 13 | `background` | 0.0417 |
| 14 | `nuclear` | 0.0408 |
| 15 | `concept` | 0.0404 |
