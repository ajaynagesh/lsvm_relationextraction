import sys;
import os;
from array import array
import time
import numpy as np
import pylab as pl
#from pylab import *
#import pylab as pl
import matplotlib.pyplot as plt
import unicodedata
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def processData(line):
    lineSplit = re.split(":| |\n|\t",line);
    validLine = False;
    for a in lineSplit:
        if(a == "Subgradient-descent"):
            #print lineSplit
            validLine = True;
    
    if(validLine):
       nums = [float(s) for s in lineSplit if is_number(s)]
       nums
       Dataset.append(nums);

def plotData():
        #print Dataset;
        xs = [x[0] for x in Dataset] 
        ys1 = [x[1] for x in Dataset] 
        ys2 = [x[2] for x in Dataset] 
        fig = plt.figure()

        fig.subplots_adjust(hspace=.5) 
       
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212);
        ax.set_xlabel('\nIterations')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        #ax1.set_xticklabels([])
        ax1.set_ylabel('Fraction of matching samples')
        ax2.set_ylabel('Objective Value')
        ax.set_title(figureNameFinal);
        ax1.scatter(xs, ys1);
        ax2.scatter(xs, ys2);
        ax1.grid();
        ax2.grid();
        plt.savefig(save_dict + "/"+figureNameFinal+'.png',dpi=300)
        plt.close();

def getOuterNumber(line):
    #print line
    lineSplit = re.split(":| |\n",line);
    #print lineSplit
    nums = [int(s) for s in lineSplit if s.isdigit()]
    #print nums;
    return nums[0]; 
    
def getInnerNumber(line):
    lineSplit = re.split(":| |\n",line);
    nums = [int(s) for s in lineSplit if s.isdigit()]
    #print "Number is " 
    print nums
    return nums[0]; 

def checkInner(line):
    lineSplit = re.split(":| |\n",line);
    for a in lineSplit:
        if(a == "ITER"):
            #print "Detected Inner";
            #print len(lineSplit);
            #print lineSplit
            if(len(lineSplit) <= 3):
                print "validated " + line
                return True;
    return False;

def checkOuter(line):
    #lineSplit = line.split(": ");
    lineSplit = re.split(":| |\n",line);
    #print len(lineSplit);
    #print lineSplit
    for a in lineSplit:
        if(a == "OUTER"):
            #print len(lineSplit);
            #print lineSplit;
            if(len(lineSplit) <= 4):
                return True;
    return False;



if (len(sys.argv) < 4):
    print "Usage python plotObjective.py <filename> <dir_to_save_path> <plot_identfier_name>";
    sys.exit()

fileName = sys.argv[1];


current_outer_iter = -1;
current_inner_iter = 0;
Dataset = [];
figureName = sys.argv[2];
figureNameFinal = "";
save_dict= sys.argv[3];
os.system("mkdir " + save_dict);
with open(fileName) as f:
    for line in f:
        if(checkOuter(line)):
            current_outer_iter+=1; 
            figureName = sys.argv[2] +'_OUTER_' + str(getOuterNumber(line));
            current_inner_iter = -1;
            print "detected OUTER";
            print line;
            
        elif(checkInner(line)):
            print "Detected Inner"
            if Dataset:
                print len(Dataset)
                plotData();
            Dataset =[];
            current_inner_iter+=1;
            figureNameFinal = figureName + '_INNER_'+ str(getInnerNumber(line));
            print figureNameFinal;
        else:
            processData(line); 
