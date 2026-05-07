# Arguments:
#    base dir
#    env
#    agent
#    line 1 file
#    line 1 title
#    line 2 file
#    line 2 title
#    ...
BASE=ARG1
ENV=ARG2
AGENT=ARG3

set style fill transparent solid 0.5

set term pngcairo size 1024,512 linewidth 3 fontscale 1.5;
set output BASE."/figures/".ENV.".".AGENT.".png";
set grid;
set key box bottom center opaque;

set xlabel "Steps";
set ylabel "Average reward";
set title ((ENV eq "bandit")                 ? "Multi-armed bandit" : \
           (ENV eq "newcomb")                ? "Newcomb's problem" : \
           (ENV eq "damascus")               ? "Death in Damascus" : \
           (ENV eq "asymmetric-damascus")    ? "Asymmetric Death in Damascus" : \
           (ENV eq "coordination")           ? "Coordination game" : \
           (ENV eq "pdbandit")               ? "Policy-dependent bandit" : \
           (ENV eq "switching")              ? "Switching bandit" : \
           ENV).", ". \
          ((AGENT eq "classical")            ? "Q-learning agent" : \
           (AGENT eq "bayesian")             ? "Bayesian agent" : \
           (AGENT eq "exp3")                 ? "EXP3 agent" : \
           (AGENT eq "experimental1")        ? "Experimental agent 1" : \
           (AGENT eq "experimental2")        ? "Experimental agent 2" : \
           (AGENT eq "experimental3")        ? "Experimental agent 3" : \
           AGENT);
set yrange [0:];
NUM_LINES=(ARGC-3)/2;
plot BASE."/outputs/".ENV.".".ARGV[4].".txt" u 1:2 w l ls 1 title "Optimal policy", \
     for [i=0:NUM_LINES-1] BASE."/outputs/".ENV.".".ARGV[4+2*i].".txt" u 1:($3+$4):($3-$4) w filledcurves ls 2 notitle, \
     for [i=0:NUM_LINES-1] BASE."/outputs/".ENV.".".ARGV[4+2*i].".txt" u 1:3 w l ls i+2 title ARGV[4+2*i+1];
