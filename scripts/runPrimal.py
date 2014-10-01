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
outFileName = "10p.admm.out"
modelResultFile = "10p.admm.model.result"
allprimals = sys.argv[1];


tempBase = "";
for i in range(1,16):
    #print i;
    #print simFrac[i-1]
    tempLogs = baseLogsDir + str(i) + "_"+str(pho[i-1]) + "_"+str(simFrac[i-1]);
    temp = str(i) + "_"+str(pho[i-1]) + "_"+str(simFrac[i-1]);
    command = 'echo "\n\n\n **************************\n'+ temp + ' Primal \n" >> ' +allprimals;
    #print command;
    os.system(command);
    command = "cat  "+tempLogs+"/"+outFileName + ' | grep "primal"  >>' + allprimals
    #print command;
    os.system(command);
