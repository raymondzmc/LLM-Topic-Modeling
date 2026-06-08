# 20 Newsgroups doc #1701 — figure-source data

- LLM: ERNIE-4.5-0.3B-PT
- DST: top-k=20, T=3.0
- BoW vocab size: 2000
- Label: 2  (comp.os.ms-windows.misc)
- Top-15 IoU(BoW, DST): 0.00

## Document

```
I've been running Dos 6 for about a month.  I was generally impressed with
the improvements:  the multiple boot configurations were great, the
new commands were nice, and DoubleSpace worked fine (twice as slow for
large data transfers, twice as fast for small with SmartDrv).

Until now.

This morning at 4 am while I was working on my research paper, I had to
reboot a hung Dos program (that did no disk i/o) from within Windows 3.1.
When my machine finished rebooting, I found my windows directory and about two
thirds of my other directories were irreversibly corrupted.

I cannot afford problems like this.  I'm returning to Dos 5.

mark

Topic:
```

## BoW target — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `twice` | 0.0606 |
| 2 | `nice` | 0.0303 |
| 3 | `slow` | 0.0303 |
| 4 | `finished` | 0.0303 |
| 5 | `disk` | 0.0303 |
| 6 | `mark` | 0.0303 |
| 7 | `great` | 0.0303 |
| 8 | `boot` | 0.0303 |
| 9 | `fast` | 0.0303 |
| 10 | `month` | 0.0303 |
| 11 | `morning` | 0.0303 |
| 12 | `commands` | 0.0303 |
| 13 | `program` | 0.0303 |
| 14 | `directory` | 0.0303 |
| 15 | `new` | 0.0303 |

## DST target (Ours) — top 15 words
| Rank | Word | Probability |
|---:|:---|---:|
| 1 | `trouble` | 0.0800 |
| 2 | `running` | 0.0736 |
| 3 | `system` | 0.0677 |
| 4 | `computer` | 0.0650 |
| 5 | `software` | 0.0610 |
| 6 | `document` | 0.0550 |
| 7 | `run` | 0.0485 |
| 8 | `documentation` | 0.0475 |
| 9 | `error` | 0.0465 |
| 10 | `hard` | 0.0465 |
| 11 | `writing` | 0.0465 |
| 12 | `file` | 0.0446 |
| 13 | `performance` | 0.0437 |
| 14 | `backup` | 0.0419 |
| 15 | `problem` | 0.0411 |
