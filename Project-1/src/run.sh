# CSE 5243: Data Mining
# Project : 1
# Author  : Rajarshi Biswas
#	    Sayam Ganguly 
# File    : run.sh
# 
# Run the runner python file with the value of 5
# Change this value to change value of K

k=0
if [ $# -eq 0 ]; then
	k=5
elif [ $# -eq 1 ]; then
	k=$1
else
   echo "Wrong number of arguments provided!!"
   exit 1
fi

python runner.py $k
