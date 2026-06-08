
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
Method &
  $C_V$ & LLM & I-RBO & Purity &
  $C_V$ & LLM & I-RBO & Purity &
  $C_V$ & LLM & I-RBO & Purity \\
\midrule

\multicolumn{13}{c}{\large \textbf{K = 25}} \\
\midrule
LDA & .336 & 2.30 & .953 & .296 & .350 & 1.99 & .984 & .416 & .331 & 2.04 & .998 & .142 \\
ProdLDA & .355 & 2.58 & .991 & .335 & .362 & 2.18 & .996 & .509 & .350 & 2.60 & .997 & .238 \\
ZeroShotTM & .354 & 2.54 & .993 & .358 & .361 & 2.42 & .995 & .552 & .362 & 2.80 & .997 & .281 \\
CombinedTM & .357 & 2.64 & .993 & .353 & .364 & 2.43 & .992 & .564 & .366 & 2.86 & .996 & .279 \\
ETM & .339 & 2.43 & .942 & .335 & .348 & 2.30 & .949 & .555 & .350 & 2.35 & .946 & .160 \\
BERTopic & .361 & 2.43 & .988 & .207 & .355 & 2.19 & .996 & .479 & .376 & 2.71 & .994 & .130 \\
ECRTM & .383 & 2.45 & \colorbox{lightblue}{\textbf{.999}} & .335 & .357 & 1.72 & \colorbox{lightblue}{\textbf{1.000}} & .409 & .367 & 1.93 & \colorbox{lightblue}{\textbf{1.000}} & .066 \\
FASTopic & .371 & 2.65 & \colorbox{lightblue}{\textbf{1.000}} & .386 & .260 & 2.07 & .525 & .529 & .363 & 2.64 & .694 & .138 \\
ERNIE-4.5-0.3B &
\colorbox{lightblue}{\textbf{.395}} & 2.88 & .995 & .502 &
\colorbox{lightblue}{\textbf{.399}} & \colorbox{lightblue}{\textbf{2.93}} & .992 & \colorbox{lightblue}{\textbf{.768}} &
\colorbox{lightblue}{\textbf{.402}} & \colorbox{lightblue}{\textbf{2.94}} & .992 & \colorbox{lightblue}{\textbf{.730}} \\
Llama-3.1-8B &
.376 & \colorbox{lightblue}{\textbf{2.94}} & .995 & \colorbox{lightblue}{\textbf{.525}} &
\colorbox{lightblue}{\textbf{.392}} & \colorbox{lightblue}{\textbf{2.91}} & .995 & .754 &
\colorbox{lightblue}{\textbf{.399}} & \colorbox{lightblue}{\textbf{2.97}} & .995 & .659 \\
Llama-3.2-1B &
\colorbox{lightblue}{\textbf{.389}} & \colorbox{lightblue}{\textbf{2.91}} & .995 & \colorbox{lightblue}{\textbf{.543}} &
\colorbox{lightblue}{\textbf{.393}} & \colorbox{lightblue}{\textbf{2.94}} & .995 & \colorbox{lightblue}{\textbf{.770}} &
\colorbox{lightblue}{\textbf{.403}} & \colorbox{lightblue}{\textbf{2.95}} & .995 & .678 \\
Qwen-3.5-0.8B & .407 & 2.90 & .986 & .520 & .411 & 2.88 & .979 & .773 & .413 & 2.94 & .989 & .779 \\
Phi-3-mini & .379 & 2.74 & .998 & .536 & .390 & 2.69 & .998 & .767 & .409 & 2.86 & .992 & .790 \\
ERNIE-4.5-0.3B + ECRTM & .416 & 2.79 & .987 & .487 & .409 & 2.82 & .984 & .749 & .413 & 2.78 & .993 & .672 \\
Llama-3.1-8B + ECRTM & .398 & 2.80 & .988 & .540 & .392 & 2.75 & .984 & .744 & .396 & 2.98 & .994 & .624 \\
Llama-3.2-1B + ECRTM & .409 & 2.85 & .988 & .567 & .398 & 2.78 & .985 & .764 & .401 & 2.90 & .994 & .625 \\
Qwen-3.5-0.8B + ECRTM & .422 & 2.82 & .986 & .547 & .408 & 2.83 & .983 & .756 & .405 & 2.90 & .992 & .794 \\
Phi-3-mini + ECRTM & .380 & 2.74 & .987 & .519 & .388 & 2.75 & .984 & .749 & .402 & 2.90 & .989 & .739 \\
ERNIE-4.5-0.3B + FASTopic & .361 & 2.42 & 1.000 & .483 & .365 & 2.26 & 1.000 & .683 & .395 & 2.61 & 1.000 & .493 \\
Llama-3.1-8B + FASTopic & .370 & 2.24 & 1.000 & .519 & .367 & 2.17 & 1.000 & .691 & .417 & 2.45 & 1.000 & .521 \\
Llama-3.2-1B + FASTopic & .342 & 2.42 & 1.000 & .515 & .359 & 2.20 & 1.000 & .694 & .418 & 2.53 & 1.000 & .522 \\
Qwen-3.5-0.8B + FASTopic & .369 & 2.42 & 1.000 & .474 & .366 & 2.10 & 1.000 & .688 & .404 & 2.66 & 1.000 & .522 \\
Phi-3-mini + FASTopic & .349 & 2.19 & 1.000 & .460 & .360 & 2.14 & 1.000 & .689 & .429 & 2.60 & 1.000 & .524 \\

\midrule
\multicolumn{13}{c}{\large \textbf{K = 50}} \\
\midrule
LDA & .346 & 2.24 & .977 & .299 & .353 & 1.94 & .993 & .425 & .346 & 2.00 & .999 & .167 \\
ProdLDA & .348 & 2.52 & .993 & .357 & .352 & 2.14 & .994 & .532 & .350 & 2.64 & .993 & .266 \\
ZeroShotTM & .357 & 2.58 & .994 & .401 & .351 & 2.31 & .994 & .575 & .365 & 2.86 & .993 & .308 \\
CombinedTM & .349 & 2.58 & .995 & .389 & .356 & 2.36 & .990 & .590 & .364 & 2.84 & .987 & .309 \\
ETM & .343 & 2.35 & .930 & .367 & .354 & 2.20 & .930 & .554 & .349 & 2.30 & .927 & .155 \\
BERTopic & .359 & 2.51 & .992 & .324 & .370 & 2.20 & .996 & .579 & .375 & 2.69 & .996 & .203 \\
ECRTM & .363 & 2.32 & .998 & .378 & .359 & 1.86 & \colorbox{lightblue}{\textbf{1.000}} & .406 & .377 & 1.93 & \colorbox{lightblue}{\textbf{1.000}} & .070 \\
FASTopic & .357 & 2.58 & \colorbox{lightblue}{\textbf{.999}} & .424 & .269 & 2.00 & .625 & .557 & .358 & 2.25 & .691 & .166 \\
ERNIE-4.5-0.3B &
\colorbox{lightblue}{\textbf{.379}} & \colorbox{lightblue}{\textbf{2.84}} & .990 & .513 &
\colorbox{lightblue}{\textbf{.388}} & \colorbox{lightblue}{\textbf{2.88}} & .989 & \colorbox{lightblue}{\textbf{.779}} &
\colorbox{lightblue}{\textbf{.395}} & 2.90 & .986 & \colorbox{lightblue}{\textbf{.732}} \\
Llama-3.1-8B &
.365 & \colorbox{lightblue}{\textbf{2.88}} & .993 & \colorbox{lightblue}{\textbf{.556}} &
\colorbox{lightblue}{\textbf{.384}} & 2.86 & .992 & \colorbox{lightblue}{\textbf{.773}} &
.382 & \colorbox{lightblue}{\textbf{2.95}} & .991 & .704 \\
Llama-3.2-1B &
\colorbox{lightblue}{\textbf{.380}} & \colorbox{lightblue}{\textbf{2.90}} & .991 & \colorbox{lightblue}{\textbf{.560}} &
\colorbox{lightblue}{\textbf{.387}} & \colorbox{lightblue}{\textbf{2.91}} & .992 & \colorbox{lightblue}{\textbf{.779}} &
\colorbox{lightblue}{\textbf{.394}} & \colorbox{lightblue}{\textbf{2.94}} & .991 & .701 \\
Qwen-3.5-0.8B & .403 & 2.85 & .980 & .541 & .398 & 2.89 & .978 & .788 & .403 & 2.88 & .981 & .789 \\
Phi-3-mini & .371 & 2.72 & .994 & .558 & .385 & 2.65 & .995 & .786 & .402 & 2.87 & .988 & .802 \\
ERNIE-4.5-0.3B + ECRTM & .407 & 2.86 & .983 & .508 & .398 & 2.81 & .982 & .757 & .414 & 2.91 & .987 & .750 \\
Llama-3.1-8B + ECRTM & .390 & 2.87 & .986 & .567 & .388 & 2.82 & .984 & .758 & .391 & 2.95 & .991 & .740 \\
Llama-3.2-1B + ECRTM & .408 & 2.86 & .980 & .572 & .398 & 2.85 & .983 & .771 & .398 & 2.94 & .992 & .719 \\
Qwen-3.5-0.8B + ECRTM & .428 & 2.86 & .972 & .551 & .406 & 2.82 & .976 & .779 & .403 & 2.89 & .985 & .789 \\
Phi-3-mini + ECRTM & .378 & 2.80 & .988 & .529 & .382 & 2.74 & .986 & .774 & .408 & 2.94 & .988 & .793 \\
ERNIE-4.5-0.3B + FASTopic & .343 & 2.28 & 1.000 & .507 & .359 & 2.10 & 1.000 & .703 & .388 & 2.27 & 1.000 & .506 \\
Llama-3.1-8B + FASTopic & .352 & 2.08 & 1.000 & .547 & .353 & 2.02 & 1.000 & .702 & .400 & 2.33 & 1.000 & .532 \\
Llama-3.2-1B + FASTopic & .336 & 2.20 & 1.000 & .536 & .357 & 2.05 & 1.000 & .699 & .392 & 2.35 & 1.000 & .528 \\
Qwen-3.5-0.8B + FASTopic & .345 & 2.17 & 1.000 & .501 & .362 & 2.00 & 1.000 & .693 & .393 & 2.28 & 1.000 & .537 \\
Phi-3-mini + FASTopic & .333 & 2.01 & 1.000 & .499 & .354 & 1.94 & 1.000 & .705 & .392 & 2.28 & 1.000 & .536 \\

\midrule
\multicolumn{13}{c}{\large \textbf{K = 75}} \\
\midrule
LDA & .341 & 2.19 & .988 & .303 & .350 & 1.95 & .995 & .453 & .354 & 2.05 & .994 & .194 \\
ProdLDA & .351 & 2.46 & .992 & .360 & .355 & 2.09 & .993 & .540 & .355 & 2.63 & .989 & .277 \\
ZeroShotTM & .351 & 2.53 & .994 & .412 & .355 & 2.25 & .993 & .575 & .363 & 2.87 & .991 & .316 \\
CombinedTM & .348 & 2.53 & .992 & .407 & .361 & 2.31 & .987 & .597 & .362 & 2.86 & .983 & .315 \\
ETM & .346 & 2.31 & .930 & .377 & .351 & 2.20 & .932 & .548 & .349 & 2.23 & .930 & .148 \\
BERTopic & .359 & 2.48 & .993 & .425 & .367 & 2.20 & .996 & .589 & .375 & 2.59 & .996 & .223 \\
ECRTM & .353 & 2.25 & \colorbox{lightblue}{\textbf{.998}} & .377 & .354 & 1.89 & \colorbox{lightblue}{\textbf{1.000}} & .397 & .375 & 1.95 & \colorbox{lightblue}{\textbf{1.000}} & .056 \\
FASTopic & .353 & 2.55 & \colorbox{lightblue}{\textbf{.999}} & .425 & .283 & 1.94 & .675 & .570 & .367 & 2.20 & .684 & .180 \\
ERNIE-4.5-0.3B &
\colorbox{lightblue}{\textbf{.377}} & \colorbox{lightblue}{\textbf{2.86}} & .990 & .528 &
\colorbox{lightblue}{\textbf{.391}} & 2.88 & .988 & \colorbox{lightblue}{\textbf{.789}} &
\colorbox{lightblue}{\textbf{.395}} & 2.91 & .983 & \colorbox{lightblue}{\textbf{.742}} \\
Llama-3.1-8B &
.360 & \colorbox{lightblue}{\textbf{2.85}} & .992 & \colorbox{lightblue}{\textbf{.574}} &
.381 & 2.87 & .992 & .781 &
.383 & \colorbox{lightblue}{\textbf{2.95}} & .990 & .717 \\
Llama-3.2-1B &
.370 & \colorbox{lightblue}{\textbf{2.89}} & .990 & \colorbox{lightblue}{\textbf{.572}} &
.385 & \colorbox{lightblue}{\textbf{2.92}} & .991 & \colorbox{lightblue}{\textbf{.788}} &
\colorbox{lightblue}{\textbf{.393}} & \colorbox{lightblue}{\textbf{2.94}} & .990 & .711 \\
Qwen-3.5-0.8B & .395 & 2.84 & .977 & .551 & .398 & 2.85 & .975 & .780 & .401 & 2.87 & .981 & .790 \\
Phi-3-mini & .367 & 2.69 & .993 & .563 & .383 & 2.58 & .993 & .793 & .402 & 2.88 & .987 & .808 \\
ERNIE-4.5-0.3B + ECRTM & .400 & 2.80 & .985 & .536 & .395 & 2.86 & .982 & .778 & .410 & 2.86 & .986 & .762 \\
Llama-3.1-8B + ECRTM & .385 & 2.83 & .986 & .560 & .383 & 2.85 & .985 & .768 & .395 & 2.94 & .990 & .754 \\
Llama-3.2-1B + ECRTM & .399 & 2.84 & .981 & .590 & .387 & 2.82 & .984 & .783 & .393 & 2.94 & .991 & .738 \\
Qwen-3.5-0.8B + ECRTM & .425 & 2.82 & .970 & .568 & .408 & 2.81 & .974 & .795 & .405 & 2.83 & .980 & .815 \\
Phi-3-mini + ECRTM & .374 & 2.80 & .990 & .542 & .382 & 2.71 & .988 & .789 & .408 & 2.93 & .988 & .815 \\
ERNIE-4.5-0.3B + FASTopic & .337 & 2.17 & 1.000 & .522 & .359 & 1.95 & 1.000 & .706 & .378 & 2.14 & 1.000 & .517 \\
Llama-3.1-8B + FASTopic & .336 & 1.99 & 1.000 & .565 & .351 & 1.93 & 1.000 & .707 & .386 & 2.21 & 1.000 & .541 \\
Llama-3.2-1B + FASTopic & .334 & 2.11 & 1.000 & .552 & .355 & 1.97 & 1.000 & .704 & .377 & 2.20 & 1.000 & .535 \\
Qwen-3.5-0.8B + FASTopic & .340 & 2.02 & 1.000 & .516 & .358 & 1.93 & 1.000 & .700 & .383 & 2.15 & 1.000 & .545 \\
Phi-3-mini + FASTopic & .333 & 1.98 & 1.000 & .510 & .350 & 1.91 & 1.000 & .711 & .384 & 2.20 & 1.000 & .542 \\

\midrule
\multicolumn{13}{c}{\large \textbf{K = 100}} \\
\midrule
LDA & .342 & 2.15 & .991 & .308 & .348 & 1.92 & .995 & .470 & .386 & 1.93 & .915 & .192 \\
ProdLDA & .349 & 2.39 & .992 & .373 & .352 & 2.05 & .992 & .552 & .353 & 2.60 & .986 & .279 \\
ZeroShotTM & .352 & 2.48 & .993 & .417 & .358 & 2.23 & .993 & .589 & .363 & 2.84 & .989 & .324 \\
CombinedTM & .351 & 2.51 & .993 & .415 & .357 & 2.35 & .983 & .599 & .363 & 2.85 & .979 & .320 \\
ETM & .347 & 2.29 & .933 & .361 & .351 & 2.20 & .931 & .548 & .345 & 2.26 & .933 & .141 \\
BERTopic & .359 & 2.48 & .993 & .454 & .364 & 2.19 & .996 & .601 & .370 & 2.53 & .997 & .252 \\
ECRTM & .343 & 2.11 & \colorbox{lightblue}{\textbf{1.000}} & .364 & .355 & 1.95 & \colorbox{lightblue}{\textbf{1.000}} & .382 & .371 & 1.98 & \colorbox{lightblue}{\textbf{1.000}} & .055 \\
FASTopic & .350 & 2.58 & .998 & .431 & .293 & 1.81 & .679 & .571 & .365 & 2.13 & .678 & .199 \\
ERNIE-4.5-0.3B &
\colorbox{lightblue}{\textbf{.373}} & \colorbox{lightblue}{\textbf{2.86}} & .989 & .536 &
\colorbox{lightblue}{\textbf{.389}} & 2.89 & .987 & .789 &
\colorbox{lightblue}{\textbf{.395}} & 2.90 & .984 & \colorbox{lightblue}{\textbf{.743}} \\
Llama-3.1-8B &
.355 & 2.84 & .992 & \colorbox{lightblue}{\textbf{.580}} &
.377 & 2.87 & .991 & .786 &
.382 & \colorbox{lightblue}{\textbf{2.96}} & .989 & .720 \\
Llama-3.2-1B &
.367 & \colorbox{lightblue}{\textbf{2.87}} & .989 & \colorbox{lightblue}{\textbf{.581}} &
.382 & \colorbox{lightblue}{\textbf{2.92}} & .990 & \colorbox{lightblue}{\textbf{.797}} &
.388 & \colorbox{lightblue}{\textbf{2.96}} & .990 & .702 \\
Qwen-3.5-0.8B & .391 & 2.84 & .976 & .555 & .397 & 2.81 & .974 & .784 & .397 & 2.86 & .979 & .795 \\
Phi-3-mini & .362 & 2.66 & .993 & .572 & .384 & 2.59 & .992 & .802 & .405 & 2.83 & .986 & .811 \\
ERNIE-4.5-0.3B + ECRTM & .393 & 2.82 & .986 & .552 & .390 & 2.80 & .984 & .786 & .405 & 2.84 & .987 & .785 \\
Llama-3.1-8B + ECRTM & .378 & 2.80 & .986 & .578 & .381 & 2.86 & .986 & .789 & .391 & 2.95 & .991 & .761 \\
Llama-3.2-1B + ECRTM & .398 & 2.81 & .979 & .599 & .388 & 2.80 & .985 & .791 & .395 & 2.89 & .991 & .751 \\
Qwen-3.5-0.8B + ECRTM & .419 & 2.80 & .970 & .578 & .403 & 2.82 & .975 & .793 & .410 & 2.84 & .977 & .823 \\
Phi-3-mini + ECRTM & .368 & 2.79 & .991 & .554 & .380 & 2.67 & .991 & .801 & .397 & 2.90 & .989 & .827 \\
ERNIE-4.5-0.3B + FASTopic & .333 & 2.08 & 1.000 & .528 & .355 & 1.85 & 1.000 & .715 & .376 & 2.08 & 1.000 & .515 \\
Llama-3.1-8B + FASTopic & .332 & 1.97 & 1.000 & .590 & .351 & 1.91 & 1.000 & .732 & .379 & 2.20 & 1.000 & .575 \\
Llama-3.2-1B + FASTopic & .334 & 2.08 & 1.000 & .559 & .355 & 1.93 & 1.000 & .714 & .373 & 2.21 & 1.000 & .534 \\
Qwen-3.5-0.8B + FASTopic & .335 & 1.97 & 1.000 & .524 & .355 & 1.86 & 1.000 & .700 & .377 & 2.08 & 1.000 & .543 \\
Phi-3-mini + FASTopic & .330 & 1.98 & 1.000 & .526 & .349 & 1.85 & 1.000 & .715 & .374 & 2.15 & 1.000 & .549 \\

\bottomrule
\end{tabular}
\end{adjustbox}
\caption{
Automatic evaluation results on the top-15 words averaged over four numbers of topics ($K=25,50,75,100$), with five random seeds per $K$.
Results not statistically significantly different from the best method ($p \ge 0.05$) under Welch's $t$-test are highlighted in blue.
Rows without a suffix use the ProdLDA backbone with our proposed method; rows suffixed with \texttt{+ ECRTM} or \texttt{+ FASTopic} use the ECRTM or FASTopic backbone.
}
\label{tab:complete-results}
\end{table*}
