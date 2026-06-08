\begin{table*}[t]
\centering
\begin{adjustbox}{width=\linewidth}
\begin{tabular}{l *{6}{c}}
\toprule
\multicolumn{1}{c}{} &
  \multicolumn{2}{c}{\large \textbf{20NewsGroup}} &
  \multicolumn{2}{c}{\large \textbf{TweetTopic}} &
  \multicolumn{2}{c}{\large \textbf{StackOverflow}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Method &
  P@5 & P@10 &
  P@5 & P@10 &
  P@5 & P@10 \\
\midrule
\multicolumn{7}{c}{\large \textbf{K = 25}} \\
\midrule
LDA & .247 & .238 & .625 & .470 & .085 & .081 \\
ProdLDA & .315 & .306 & .732 & .599 & .224 & .214 \\
ZeroShotTM & .349 & .341 & .732 & .623 & .254 & .246 \\
CombinedTM & .334 & .328 & .673 & .599 & .236 & .229 \\
ETM & .298 & .290 & .797 & .639 & .137 & .127 \\
BERTopic & .390 & .380 & .473 & .431 & .294 & .283 \\
ECRTM & .320 & .309 & .745 & .547 & .094 & .084 \\
FASTopic & .391 & .378 & .792 & .635 & .146 & .136 \\
ERNIE-4.5-0.3B & .471 & .465 & .856 & .821 & .725 & .718 \\
Llama-3.1-8B & .526 & .519 & .874 & .834 & .720 & .711 \\
Llama-3.2-1B & .516 & .510 & .867 & .828 & .711 & .703 \\
Qwen-3.5-0.8B & .483 & .477 & .859 & .826 & .787 & .782 \\
Phi-3-mini & .528 & .522 & .858 & .824 & .812 & .806 \\
ERNIE-4.5-0.3B + ECRTM & .552 & .533 & .924 & .859 & .786 & .774 \\
Llama-3.1-8B + ECRTM & \colorbox{lightblue}{\textbf{.629}} & \colorbox{lightblue}{\textbf{.609}} & \colorbox{lightblue}{\textbf{.935}} & \colorbox{lightblue}{\textbf{.878}} & .823 & .808 \\
Llama-3.2-1B + ECRTM & .593 & .578 & .927 & .864 & .770 & .755 \\
Qwen-3.5-0.8B + ECRTM & .551 & .538 & .923 & .858 & .829 & .822 \\
Phi-3-mini + ECRTM & .617 & .597 & .932 & \colorbox{lightblue}{\textbf{.876}} & \colorbox{lightblue}{\textbf{.859}} & \colorbox{lightblue}{\textbf{.851}} \\
ERNIE-4.5-0.3B + FASTopic & .360 & .364 & .584 & .584 & .383 & .376 \\
Llama-3.1-8B + FASTopic & .420 & .421 & .594 & .588 & .384 & .387 \\
Llama-3.2-1B + FASTopic & .404 & .407 & .592 & .592 & .401 & .397 \\
Qwen-3.5-0.8B + FASTopic & .364 & .362 & .591 & .587 & .403 & .405 \\
Phi-3-mini + FASTopic & .358 & .355 & .595 & .593 & .416 & .414 \\
\midrule
\multicolumn{7}{c}{\large \textbf{K = 50}} \\
\midrule
LDA & .251 & .237 & .671 & .497 & .106 & .102 \\
ProdLDA & .346 & .335 & .774 & .623 & .246 & .234 \\
ZeroShotTM & .385 & .375 & .779 & .649 & .279 & .268 \\
CombinedTM & .367 & .358 & .746 & .638 & .263 & .255 \\
ETM & .320 & .313 & .790 & .642 & .138 & .127 \\
BERTopic & .412 & .399 & .559 & .535 & .304 & .295 \\
ECRTM & .348 & .336 & .745 & .546 & .107 & .097 \\
FASTopic & .421 & .406 & .810 & .665 & .184 & .174 \\
ERNIE-4.5-0.3B & .507 & .497 & .908 & .847 & .756 & .748 \\
Llama-3.1-8B & .583 & .571 & .914 & .856 & .802 & .791 \\
Llama-3.2-1B & .554 & .544 & .910 & .848 & .758 & .747 \\
Qwen-3.5-0.8B & .519 & .511 & .911 & .851 & .808 & .801 \\
Phi-3-mini & .565 & .555 & .905 & .848 & .837 & .831 \\
ERNIE-4.5-0.3B + ECRTM & .560 & .542 & .924 & .859 & .804 & .794 \\
Llama-3.1-8B + ECRTM & \colorbox{lightblue}{\textbf{.640}} & \colorbox{lightblue}{\textbf{.621}} & \colorbox{lightblue}{\textbf{.935}} & \colorbox{lightblue}{\textbf{.878}} & .848 & .836 \\
Llama-3.2-1B + ECRTM & .597 & .583 & .926 & .863 & .809 & .797 \\
Qwen-3.5-0.8B + ECRTM & .552 & .540 & .925 & .861 & .834 & .827 \\
Phi-3-mini + ECRTM & .628 & .607 & .933 & \colorbox{lightblue}{\textbf{.877}} & \colorbox{lightblue}{\textbf{.867}} & \colorbox{lightblue}{\textbf{.860}} \\
ERNIE-4.5-0.3B + FASTopic & .413 & .411 & .614 & .611 & .395 & .394 \\
Llama-3.1-8B + FASTopic & .452 & .448 & .621 & .617 & .412 & .407 \\
Llama-3.2-1B + FASTopic & .443 & .444 & .605 & .605 & .404 & .403 \\
Qwen-3.5-0.8B + FASTopic & .408 & .408 & .611 & .603 & .414 & .413 \\
Phi-3-mini + FASTopic & .418 & .416 & .626 & .617 & .461 & .454 \\
\midrule
\multicolumn{7}{c}{\large \textbf{K = 75}} \\
\midrule
LDA & .244 & .227 & .696 & .513 & .128 & .124 \\
ProdLDA & .357 & .343 & .779 & .625 & .251 & .239 \\
ZeroShotTM & .396 & .384 & .792 & .655 & .289 & .277 \\
CombinedTM & .378 & .369 & .771 & .651 & .268 & .259 \\
ETM & .320 & .314 & .782 & .637 & .135 & .126 \\
BERTopic & .424 & .413 & .572 & .551 & .310 & .301 \\
ECRTM & .351 & .338 & .751 & .551 & .123 & .109 \\
FASTopic & .432 & .416 & .819 & .682 & .208 & .196 \\
ERNIE-4.5-0.3B & .524 & .513 & .918 & .854 & .764 & .755 \\
Llama-3.1-8B & .596 & .583 & .923 & .861 & .806 & .793 \\
Llama-3.2-1B & .566 & .555 & .917 & .852 & .771 & .760 \\
Qwen-3.5-0.8B & .528 & .520 & .918 & .855 & .811 & .805 \\
Phi-3-mini & .576 & .564 & .917 & .857 & .843 & .836 \\
ERNIE-4.5-0.3B + ECRTM & .561 & .545 & .923 & .858 & .802 & .792 \\
Llama-3.1-8B + ECRTM & \colorbox{lightblue}{\textbf{.634}} & \colorbox{lightblue}{\textbf{.614}} & \colorbox{lightblue}{\textbf{.935}} & \colorbox{lightblue}{\textbf{.877}} & .849 & .838 \\
Llama-3.2-1B + ECRTM & .593 & .579 & .924 & .861 & .811 & .799 \\
Qwen-3.5-0.8B + ECRTM & .555 & .544 & .925 & .862 & .836 & .829 \\
Phi-3-mini + ECRTM & .626 & .605 & \colorbox{lightblue}{\textbf{.934}} & \colorbox{lightblue}{\textbf{.878}} & \colorbox{lightblue}{\textbf{.871}} & \colorbox{lightblue}{\textbf{.865}} \\
ERNIE-4.5-0.3B + FASTopic & .428 & .425 & .623 & .617 & .403 & .399 \\
Llama-3.1-8B + FASTopic & .473 & .471 & .628 & .623 & .408 & .408 \\
Llama-3.2-1B + FASTopic & .461 & .458 & .619 & .611 & .415 & .411 \\
Qwen-3.5-0.8B + FASTopic & .429 & .428 & .624 & .614 & .430 & .426 \\
Phi-3-mini + FASTopic & .434 & .430 & .638 & .625 & .465 & .458 \\
\midrule
\multicolumn{7}{c}{\large \textbf{K = 100}} \\
\midrule
LDA & .236 & .220 & .705 & .521 & .133 & .129 \\
ProdLDA & .366 & .353 & .782 & .629 & .248 & .236 \\
ZeroShotTM & .404 & .391 & .799 & .660 & .296 & .283 \\
CombinedTM & .383 & .373 & .785 & .660 & .271 & .262 \\
ETM & .309 & .302 & .775 & .632 & .130 & .120 \\
BERTopic & .427 & .416 & .582 & .558 & .316 & .307 \\
ECRTM & .347 & .333 & \colorbox{lightblue}{\textbf{.606}} & .475 & .131 & .115 \\
FASTopic & .441 & .423 & .823 & .687 & .219 & .206 \\
ERNIE-4.5-0.3B & .533 & .520 & .919 & .854 & .770 & .762 \\
Llama-3.1-8B & .603 & .589 & .925 & .864 & .807 & .794 \\
Llama-3.2-1B & .575 & .563 & .921 & .856 & .768 & .755 \\
Qwen-3.5-0.8B & .534 & .525 & .920 & .856 & .814 & .807 \\
Phi-3-mini & .584 & .570 & .922 & .861 & .843 & .836 \\
ERNIE-4.5-0.3B + ECRTM & .564 & .547 & .923 & .859 & .806 & .797 \\
Llama-3.1-8B + ECRTM & \colorbox{lightblue}{\textbf{.632}} & \colorbox{lightblue}{\textbf{.612}} & \colorbox{lightblue}{\textbf{.934}} & \colorbox{lightblue}{\textbf{.877}} & .848 & .836 \\
Llama-3.2-1B + ECRTM & .593 & .578 & .923 & .859 & .812 & .801 \\
Qwen-3.5-0.8B + ECRTM & .558 & .547 & .926 & .863 & .836 & .829 \\
Phi-3-mini + ECRTM & .625 & .604 & \colorbox{lightblue}{\textbf{.934}} & \colorbox{lightblue}{\textbf{.878}} & \colorbox{lightblue}{\textbf{.871}} & \colorbox{lightblue}{\textbf{.864}} \\
ERNIE-4.5-0.3B + FASTopic & .434 & .428 & .642 & .634 & .400 & .398 \\
Llama-3.1-8B + FASTopic & .493 & .492 & .665 & .660 & .459 & .454 \\
Llama-3.2-1B + FASTopic & .471 & .468 & .633 & .629 & .418 & .412 \\
Qwen-3.5-0.8B + FASTopic & .439 & .436 & .629 & .617 & .436 & .429 \\
Phi-3-mini + FASTopic & .450 & .446 & .643 & .630 & .469 & .461 \\
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{
Complete retrieval evaluation (P@5, P@10) for each number of topics ($K=25, 50, 75, 100$), each cell averaged over 5 random seeds.
For each $(K, \text{dataset}, \text{metric})$ column, methods are compared using Welch's independent samples $t$-test ($\alpha=0.05$). The best result and those not statistically significantly different from it ($p \ge 0.05$) are \colorbox{lightblue}{\textbf{highlighted in blue}}.
Rows without a suffix use the ProdLDA backbone with our proposed method; rows suffixed with \texttt{+ ECRTM} or \texttt{+ FASTopic} use the ECRTM or FASTopic backbone.
}
\label{tab:complete-retrieval-results}
\end{table*}
