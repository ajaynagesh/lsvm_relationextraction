

1. Set the mosek environment using the command : 
   export MOSEKLM_LICENSE_FILE=~/Research/software/mosek.5/mosek.lic

2. Check the last date of expiration of the mosek licence in the mosek.lic. This is
   especially important when the weights are not being updated (they remain to be 0). 
   Also in the log file, check the output of cut_error[i], if it is 0 consecutively. 
   Request for a new licence if expired.
