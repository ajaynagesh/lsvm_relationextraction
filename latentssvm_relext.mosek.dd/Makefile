# Makefile for Latent Structural SVM

CC=gcc
CFLAGS= -g -Wall
#CFLAGS= -O3 -fomit-frame-pointer -ffast-math
#CFLAGS = -O3 -pg
LD=gcc
#LDFLAGS= -g
LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm
MOSEK_H= /home/ajay/Research/software/mosek.5/5/tools/platform/linux64x86/h/
MSKLINKFLAGS= -lirc -lguide -limf -lsvml -lunwind -lmosek64 -lpthread -lc -ldl -lm
MSKLIBPATH= /home/ajay/Research/software/mosek.5/5/tools/platform/linux64x86/bin/

all: svm_latent_learn_online svm_latent_classify
	cd java && ant

all_naacl: svm_latent_learn svm_latent_classify
	cd java && ant

clean: 
	rm -f *.o
	rm -f svm_latent_learn svm_latent_classify
	cd java && ant clean

svm_latent_learn_online: svm_struct_latent_cccp_online.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o 
	$(LD) $(LDFLAGS) svm_struct_latent_cccp_online.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o -o svm_latent_learn_online -L $(MSKLIBPATH) $(MSKLINKFLAGS)

svm_latent_learn: svm_struct_latent_cccp.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o 
	$(LD) $(LDFLAGS) svm_struct_latent_cccp.o svm_common.o mosek_qp_optimize.o svm_struct_latent_api.o -o svm_latent_learn -L $(MSKLIBPATH) $(MSKLINKFLAGS)

svm_latent_classify: svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o 
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_common.o svm_struct_latent_api.o -o svm_latent_classify $(LIBS)

svm_struct_latent_cccp_online.o: svm_struct_latent_cccp_online.c
	$(CC) -c $(CFLAGS) svm_struct_latent_cccp_online.c -o svm_struct_latent_cccp_online.o

svm_struct_latent_cccp.o: svm_struct_latent_cccp.c
	$(CC) -c $(CFLAGS) svm_struct_latent_cccp.c -o svm_struct_latent_cccp.o

svm_common.o: ./svm_light/svm_common.c ./svm_light/svm_common.h ./svm_light/kernel.h
	$(CC) -c $(CFLAGS) ./svm_light/svm_common.c -o svm_common.o

mosek_qp_optimize.o: mosek_qp_optimize.c
	$(CC) -c $(CFLAGS) mosek_qp_optimize.c -o mosek_qp_optimize.o -I $(MOSEK_H)

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o
