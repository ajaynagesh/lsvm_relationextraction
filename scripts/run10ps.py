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

count = sys.argv[1];

baseTempDir = "tmp"
baseStatsName = "dataset/stats/train-riedel-10p."
baseDataSet = "dataset/10pdataset/train-riedel-10p."
baseLogsDir = "logs_"
outFileName = "10p.admm.out"
errFileName = "10p.admm.err"
modelFileName = "10p.admm.model"


tempBase = "";
for i in range(1,16):
    print i;
    print simFrac[i-1]
    tempBase = "tmp" +  str(i)+ "/";
    tempStats = baseStatsName +str(i) +".stats";
    tempDataset = baseDataSet+str(i)+".data";
    tempLogs = baseLogsDir + str(i) + "_"+str(pho[i-1]) + "_"+str(simFrac[i-1]);
    os.system("mkdir " + tempLogs);
    os.system("mkdir " + tempBase);
    command = "nohup ./svm_latent_learn -f " +tempBase+" -w 0.5  -y "+str(simFrac[i-1])+" -z "+tempStats+ " -o "+str(pho[i-1])+" " + tempDataset+" " + tempLogs+"/"+modelFileName+ " > "+tempLogs+"/"+outFileName+ " 2> " +tempLogs+"/"+errFileName +" &"
    print command;
    os.system(command);
