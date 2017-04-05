# Project 1
# File  : runner.py
# Author: Rajarshi Biswas
#       : Sayam Ganguly
import sys
from Iris import analyze_iris_data
from Income import analyze_income_data

k = int(sys.argv[1])
# Analyze the Iris data
analyze_iris_data(k)
# Analyze the Income data
analyze_income_data(k)
