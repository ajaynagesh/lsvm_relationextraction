
import os; import sys;
import os;
from array import array
import time
#from pylab import *
#import pylab as pl



if (len(sys.argv) < 2):
    print "Usage python run10ps.py <count>";
    sys.exit()

pho = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4];
simFrac = [0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98,0.98];

baseLogsDir = "logs_"
modelFileName = "10p.admm.model"
modelResultFile = "10p.admm.model.result"
allresults = sys.argv[1];


tempBase = "";
for i in range(1,16):
    #print i;
    #print simFrac[i-1]
    tempLogs = baseLogsDir + str(i) + "_"+str(pho[i-1]) + "_"+str(simFrac[i-1]);
    temp = str(i) + "_"+str(pho[i-1]) + "_"+str(simFrac[i-1]);
    command = "java -Xmx6G -cp java/bin/:java/lib/* evaluation.ClassifyStructEgAll "+ tempLogs+"/"+modelFileName +"  dataset/testSVM.pos_r.data dataset/reidel_mapping" 


    #print command;
    os.system(command);
    command = 'echo "\n\n\n **************************\n'+ temp + ' results " >> ' +allresults;
    #print command;
    os.system(command);
    command = "java -Xmx6G -cp java/bin/:java/lib/* evaluation.ReidelEval "+tempLogs+"/"+modelResultFile + " 0.5 " ">>" + allresults
    #print command;
    os.system(command);
