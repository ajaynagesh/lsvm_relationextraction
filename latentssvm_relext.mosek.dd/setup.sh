#./svm_latent_learn_online -n 4 -f tmp3/ -w 0.5 -y 0.9 -z dataset/v.small.stats -o 0.0 -a 0 -b 1 -C 5 -c 5 -e 0.5 dataset/v.small.data temp-x.model >> abc.txt 2>>abc.txt & -- command to run online svm pre-version -- working without segfaults -- java code to split datasets
#./svm_latent_learn_online -n 4 -f tmp3/ -w 0.5 -y 0.9 -z dataset/v.small.stats -o 0.0 -a 0 -b 1 -C 5 -c 5 -e 0.5 -K 5 -E 2 dataset/v.small.data temp-y.model >> cdf.txt 2>>cdf.txt & -- online svm-version -- running without faults on the local m/c on a small dataset (change of vars, augmenting loss multiple epochs)

export MOSEKLM_LICENSE_FILE=~/Research/software/mosek.5/mosek.lic
export LD_LIBRARY_PATH='/home/ajay/Research/software/lp_solve/:/home/ajay/Research/software/mosek.5/5/tools/platform/linux64x86/bin/:/home/ajay/Research/software/x86-64_sles10_4.1/'
ulimit -c unlimited

