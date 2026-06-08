\begin{table*}[ht]
\centering
\begin{adjustbox}{width=\linewidth}
\begin{tabular}{l *{12}{c}}
\toprule
\multicolumn{1}{c}{} &
  \multicolumn{4}{c}{\large \textbf{20NewsGroup}} &
  \multicolumn{4}{c}{\large \textbf{TweetTopic}} &
  \multicolumn{4}{c}{\large \textbf{StackOverflow}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}
\multicolumn{1}{c}{} &
  \texttt{\small $C_V$} &
  \texttt{\small LLM} &
  \texttt{\small I-RBO} &
  \texttt{\small Purity} &
  \texttt{\small $C_V$} &
  \texttt{\small LLM} &
  \texttt{\small I-RBO} &
  \texttt{\small Purity} &
  \texttt{\small $C_V$} &
  \texttt{\small LLM} &
  \texttt{\small I-RBO} &
  \texttt{\small Purity} \\
\midrule
LDA &
.341 & 2.22 & .977 & .301 &
.350 & 1.95 & .992 & .441 &
.354 & 2.00 & .977 & .174 \\

ProdLDA &
.351 & 2.49 & .992 & .356 &
.355 & 2.11 & .994 & .533 &
.352 & 2.62 & .991 & .265 \\

CombinedTM &
.351 & 2.56 & .993 & .391 &
.360 & 2.36 & .988 & .588 &
.364 & 2.85 & .986 & .306 \\

ZeroshotTM &
.354 & 2.53 & .994 & .397 &
.356 & 2.30 & .994 & .573 &
.363 & 2.84 & .993 & .307 \\

ETM &
.344 & 2.35 & .934 & .360 &
.351 & 2.22 & .936 & .552 &
.348 & 2.29 & .934 & .151 \\

BERTopic &
.360 & 2.48 & .992 & .352 &
.364 & 2.19 & .996 & .562 &
.374 & 2.63 & .996 & .202 \\

ECRTM &
.360 & 2.28 & .999 & .364 &
.356 & 1.85 & 1.000 & .399 &
.373 & 1.95 & 1.000 & .062 \\

FASTopic &
.358 & 2.59 & .999 & .416 &
.276 & 1.96 & .626 & .557 &
.363 & 2.30 & .687 & .171 \\

\midrule
\multicolumn{13}{c}{\large Ours: ProdLDA + DSL} \\
\midrule
\texttt{ERNIE-4.5-0.3B} &
.381 & 2.86 & .991 & .520 &
.392 & 2.90 & .989 & .781 &
.397 & 2.91 & .986 & .737 \\

\texttt{Qwen3.5-0.8B} &
.399 & 2.86 & .980 & .542 &
.401 & 2.86 & .976 & .781 &
.403 & 2.89 & .983 & .788 \\

\texttt{Llama-3.2-1B} &
.377 & 2.89 & .991 & .564 &
.387 & 2.92 & .992 & .784 &
.395 & 2.95 & .991 & .698 \\

\midrule
\multicolumn{13}{c}{\large Ours: FASTopic + DSL} \\
\midrule
\texttt{ERNIE-4.5-0.3B} &
.344 & 2.24 & 1.000 & .510 &
.359 & 2.04 & 1.000 & .702 &
.384 & 2.27 & 1.000 & .508 \\

\texttt{Qwen3.5-0.8B} &
.347 & 2.15 & 1.000 & .504 &
.360 & 1.97 & 1.000 & .695 &
.389 & 2.29 & 1.000 & .537 \\

\texttt{Llama-3.2-1B} &
.337 & 2.20 & 1.000 & .541 &
.357 & 2.03 & 1.000 & .703 &
.390 & 2.32 & 1.000 & .530 \\

\midrule
\multicolumn{13}{c}{\large Ours: ECRTM + DSL} \\
\midrule
\texttt{ERNIE-4.5-0.3B} &
.404 & 2.82 & .985 & .521 &
.398 & 2.82 & .983 & .767 &
.410 & 2.85 & .988 & .742 \\

\texttt{Qwen3.5-0.8B} &
.423 & 2.82 & .975 & .561 &
.406 & 2.82 & .977 & .781 &
.406 & 2.86 & .984 & .805 \\

\texttt{Llama-3.2-1B} &
.404 & 2.84 & .982 & .582 &
.393 & 2.81 & .984 & .777 &
.397 & 2.92 & .992 & .708 \\

\bottomrule
\end{tabular}
\end{adjustbox}
\caption{
Automatic evaluation results on the top-15 words averaged over 4 numbers of topics ($K$ = 25, 50, 75, 100), where results for each $K$ are averaged over 5 random seeds.
For each dataset and metric, methods are compared using Welch's independent samples $t$-test \citep{welch1947generalization} with $\alpha=0.05$.
The best-performing method, as well as results that are not statistically significantly different from it ($p \ge 0.05$), are \colorbox{lightblue}{\textbf{highlighted in blue}}.
}
\label{tab:main-results}
\vspace{-1em}
\end{table*}
