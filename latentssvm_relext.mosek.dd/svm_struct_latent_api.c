/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct                     */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 17.Dec.08                                                    */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include "svm_struct_latent_api_types.h"

void test_print(SAMPLE sample){
	printf("%d\n",sample.n);
	int i;
	for(i = 0; i < sample.n; i ++){
		EXAMPLE e = (EXAMPLE)sample.examples[i];
		int num_rels = e.y.num_relations;
		printf("%d\n",num_rels);
		int j;
		for(j = 0; j < num_rels; j ++){
			printf("%d\n",e.y.relations[j]);
		}
		int num_mentions = e.x.num_mentions;
		printf("%d\n", num_mentions);
		int k;
		for(k = 0; k < num_mentions; k ++){

			WORD *mention_features = (WORD*) e.x.mention_features[k].words;
			int f_sz = atoi(e.x.mention_features[k].userdefined);

			int l;
			printf("%d\t",f_sz);
			for(l = 0; l < f_sz; l ++){
				printf("%d:%.1f ",mention_features[l].wnum, mention_features[l].weight);
			}
			printf("\n");
		}
	}
}

SAMPLE read_struct_examples_chunk(char *file) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialised in this function or by calling init_latent_variables().
*/
	SAMPLE sample;

	FILE *fp = fopen(file,"r");
	if (fp==NULL) {
		printf("Cannot open input file %s!\n", file);
		exit(1);
	}

	long int num_egs = 0, eg_id;
	int num_mentions, num_rels, total_num_rels;

	fscanf(fp, "%ld\n", &num_egs);// --> no. of entity pairs (egs)
	//printf("Number of examples : %ld\n", num_egs);

	fscanf(fp, "%d\n", &total_num_rels); // --> Total number of relation labels
	//printf("Total number of relation labels: %d\n", total_num_rels);

	// init. 'SAMPLE'
	sample.n = num_egs;
	sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_egs);

	for(eg_id = 0; eg_id < num_egs; eg_id ++){
		//printf("----\nEg : %ld\n", eg_id);

		//init 'EXAMPLE'
		EXAMPLE *e = &(sample.examples[eg_id]);

		fscanf(fp, "%d\n", &num_rels); // --> eg. i -- no. of relation labels
		//printf("Num_relations %d\n", num_rels);

		//init 'LABEL' (e->y)
		e->y.num_relations = num_rels;

		if(num_rels > 0) {
			int y[num_rels];
			int yid;
			for(yid = 0; yid < num_rels; yid++){
				fscanf(fp, "%d\n", (y+yid)); // --> eg. i -- relation label
			}
			e->y.relations = (int*)malloc(sizeof(int)*num_rels);
			for(yid = 0; yid < num_rels; yid++){
				//printf("%d ",y[yid]);
				e->y.relations[yid] = y[yid];
			}
			//printf("\n");
		}

		fscanf(fp, "%d\n", &num_mentions); // --> eg. i -- no. of mention labels
		//printf("Num_mentions %d\n", num_mentions);

		// init 'PATTERN' (e->x)
		e->x.num_mentions = num_mentions;
		e->x.mention_features = (SVECTOR*)malloc(sizeof(SVECTOR)*num_mentions);

		// init 'LATENT_VAR' (e->h)
		e->h.num_mentions = num_mentions;
		// Each of the mention labels should be initialized to nil label
		// But we do not have a specific nil label index now.
		// Right now initialising to 0 (nillabel)
		e->h.mention_labels = (int*) malloc(sizeof(int)*num_mentions);
		int i;
		for(i = 0; i < num_mentions; i ++){
			e->h.mention_labels[i] = 0;
		}

		int m;
		for(m = 0; m < num_mentions; m++){
			int f_sz;
			fscanf(fp, "%d\t", &f_sz); // --> eg. i, men m -- sz of the Fvector
			//printf("(sz:%d)\t",f_sz);
			char * f_sz_str = (char*)malloc(10);
			sprintf(f_sz_str,"%d", f_sz);
			e->x.mention_features[m].userdefined = f_sz_str;

			e->x.mention_features[m].words = (WORD*)malloc(sizeof(WORD)*(f_sz + 1));

			int i;
			for(i = 0; i < f_sz; i ++){
				int f_id; float f_val;
				fscanf(fp, "%d:%f ", &f_id, &f_val); // --> eg. i, men m -- Fvector (<fid:freq> <fid:freq> ....)
				//printf("%d:%.1f ", f_id, f_val);

				e->x.mention_features[m].words[i].wnum = f_id;
				e->x.mention_features[m].words[i].weight = f_val;
			}

			// Add 0 to the last word ... might be necessary somewhere
			e->x.mention_features[m].words[i].wnum = 0;
			e->x.mention_features[m].words[i].weight = 0;

			//printf("\n");
		}
	}

	fclose(fp);

	return(sample);
}

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) {
/*
  Read input examples {(x_1,y_1),...,(x_n,y_n)} from file.
  The type of pattern x and label y has to follow the definition in 
  svm_struct_latent_api_types.h. Latent variables h can be either
  initialised in this function or by calling init_latent_variables().
*/
	SAMPLE sample;

	FILE *fp = fopen(file,"r");
	if (fp==NULL) {
		printf("Cannot open input file %s!\n", file);
		exit(1);
	}

	long int num_egs = 0, eg_id;
	int num_mentions, num_rels, total_num_rels;
	sparm->max_feature_key=0;

	fscanf(fp, "%ld\n", &num_egs);// --> no. of entity pairs (egs)
	//printf("Number of examples : %ld\n", num_egs);

	fscanf(fp, "%d\n", &total_num_rels); // --> Total number of relation labels
	//printf("Total number of relation labels: %d\n", total_num_rels);

	// init. 'SAMPLE'
	sample.n = num_egs;
	sample.examples = (EXAMPLE*)malloc(sizeof(EXAMPLE)*num_egs);

	for(eg_id = 0; eg_id < num_egs; eg_id ++){
		//printf("----\nEg : %ld\n", eg_id);

		//init 'EXAMPLE'
		EXAMPLE *e = &(sample.examples[eg_id]);

		fscanf(fp, "%d\n", &num_rels); // --> eg. i -- no. of relation labels
		//printf("Num_relations %d\n", num_rels);

		//init 'LABEL' (e->y)
		e->y.num_relations = num_rels;

		if(num_rels > 0) {
			int y[num_rels];
			int yid;
			for(yid = 0; yid < num_rels; yid++){
				fscanf(fp, "%d\n", (y+yid)); // --> eg. i -- relation label
			}
			e->y.relations = (int*)malloc(sizeof(int)*num_rels);
			for(yid = 0; yid < num_rels; yid++){
				//printf("%d ",y[yid]);
				e->y.relations[yid] = y[yid];
			}
			//printf("\n");
		}

		fscanf(fp, "%d\n", &num_mentions); // --> eg. i -- no. of mention labels
		//printf("Num_mentions %d\n", num_mentions);

		// init 'PATTERN' (e->x)
		e->x.num_mentions = num_mentions;
		e->x.mention_features = (SVECTOR*)malloc(sizeof(SVECTOR)*num_mentions);

		// init 'LATENT_VAR' (e->h)
		e->h.num_mentions = num_mentions;
		// Each of the mention labels should be initialized to nil label
		// But we do not have a specific nil label index now.
		// Right now initialising to 0 (nillabel)
		e->h.mention_labels = (int*) malloc(sizeof(int)*num_mentions);
		int i;
		for(i = 0; i < num_mentions; i ++){
			e->h.mention_labels[i] = 0;
		}

		int m;
		for(m = 0; m < num_mentions; m++){
			int f_sz;
			fscanf(fp, "%d\t", &f_sz); // --> eg. i, men m -- sz of the Fvector
			//printf("(sz:%d)\t",f_sz);
			char * f_sz_str = (char*)malloc(10);
			sprintf(f_sz_str,"%d", f_sz);
			e->x.mention_features[m].userdefined = f_sz_str;

			e->x.mention_features[m].words = (WORD*)malloc(sizeof(WORD)*(f_sz + 1));

			int i;
			for(i = 0; i < f_sz; i ++){
				int f_id; float f_val;
				fscanf(fp, "%d:%f ", &f_id, &f_val); // --> eg. i, men m -- Fvector (<fid:freq> <fid:freq> ....)
				//printf("%d:%.1f ", f_id, f_val);

				e->x.mention_features[m].words[i].wnum = f_id;
				e->x.mention_features[m].words[i].weight = f_val;

				if (f_id>sparm->max_feature_key) sparm->max_feature_key = f_id;
			}

			// Add 0 to the last word ... might be necessary somewhere
			e->x.mention_features[m].words[i].wnum = 0;
			e->x.mention_features[m].words[i].weight = 0;


			//printf("\n");
		}
	}

	sparm->total_number_rels = total_num_rels;

	fclose(fp);

	return(sample);
}

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {
/*
  Initialize parameters in STRUCTMODEL sm. Set the diminension 
  of the feature space sm->sizePsi. Can also initialize your own
  variables in sm here. 
*/
	// To include the nil label and gap in each w chunk for a relation (w starts from 0 but svector starts from 1)
	sm->sizePsi = sparm->max_feature_key * (sparm->total_number_rels + 1);
	printf("Max feature index %ld\n", sparm->max_feature_key);
	printf("Size of w vector %ld\n",sm->sizePsi);

	printf("Number of epochs (OnlineSVM Learning) : %d\n", lparm->totalEpochs);
	printf("Number of chunks (OnlineSVM Learning) : %d\n", lparm->numChunks);

}

/* void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
* /

  /* your code here * /
}*/

void psi_write_to_file(PATTERN x, LATENT_VAR h, long max_feature_key){

	FILE *fp = fopen("tmpfiles/example","w");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", "example");
		exit(1);
	}

	int i;

	// Latent labels
	for(i = 0; i < h.num_mentions; i ++){
		fprintf(fp,"%d ", h.mention_labels[i]);
	}

	fprintf(fp, "\n");

	// Max feature key
	fprintf(fp,"%ld\n", max_feature_key);

	// Num of mentions
	fprintf(fp, "%d\n",x.num_mentions);

	// Mention features
	for(i = 0; i < x.num_mentions; i ++){
		WORD *mention_features = (WORD*) x.mention_features[i].words;
		int f_sz = atoi(x.mention_features[i].userdefined);

		int l;
		//printf("%d\t",f_sz);
		for(l = 0; l < f_sz; l ++){
			fprintf(fp, "%d:%.1f ",mention_features[l].wnum, mention_features[l].weight);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
}

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
	SVECTOR *fvec, *old_fvec;
	WORD empty[2];
	empty[0].wnum = empty[0].weight = 0;

	fvec = create_svector(empty,"",1.0);

	int i;
	for(i = 0; i < x.num_mentions; i ++){

		old_fvec = fvec;

		SVECTOR x_i = x.mention_features[i];
		int latent_label = h.mention_labels[i];

		int f_sz = atoi(x_i.userdefined);
		WORD *modified_f_list = (WORD*)malloc(sizeof(WORD) * (f_sz+1));
		int j;
		for(j = 0; j < f_sz; j ++){
			modified_f_list[j].wnum = (x_i.words[j].wnum)+ (latent_label*sparm->max_feature_key);
			modified_f_list[j].weight = x_i.words[j].weight;
		}
		modified_f_list[j].wnum = 0;
		modified_f_list[j].weight = 0;

		SVECTOR *x_i_new = create_svector(modified_f_list,"",1.0);
		free(modified_f_list);

		fvec = add_ss(x_i_new, old_fvec);
		free(x_i_new);
		free(old_fvec);

	}

//	while(fvec->words->wnum){
//		printf("%d:%.1f ",fvec->words->wnum, fvec->words->weight);
//		fvec->words++;
//	}
//	printf("\n");
//	printf("Done printing .. return\n");

	return fvec;
}

//SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
///*
//  Creates the feature vector \Psi(x,y,h) and return a pointer to
//  sparse vector SVECTOR in SVM^light format. The dimension of the
//  feature vector returned has to agree with the dimension in sm->sizePsi.
//*/
//	SVECTOR *fvec;
//
//	// Write the value of PATTERN x and LATENT_VAR h to a temporary file
//	psi_write_to_file(x, h, sparm->max_feature_key);
//
//	// CALL the method psiHelper FROM JAVA
//	system("java -cp \'java/bin:java/lib/*\' javaHelpers.psiHelper tmpfiles/example");
//
//	// Read the SVECTOR from the file written by Java
//	FILE *fp = fopen("tmpfiles/example.svector","r");
//	if (fp==NULL) {
//		printf("Cannot open input file %s!\n", "example.svector");
//		exit(1);
//	}
//	int svector_sz = -1;
//	fscanf(fp, "%d\t", &svector_sz);
//
//	int i;
//	WORD *words = malloc(sizeof(WORD)*(svector_sz + 1));
//	for(i = 0; i < svector_sz; i ++){
//		int fid; float freq;
//		fscanf(fp,"%d:%f ",&fid, &freq);
//		words[i].wnum = fid;
//		words[i].weight = freq;
//	}
//
//	words[i].wnum = 0;
//	words[i].weight = 0;
//	fvec = create_svector(words,"",1.0);
//
//	// CLEAN UP
//	free(words);
//	fclose(fp);
//	//delete the temporary files
//
//	return(fvec);
//}

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Makes prediction with input pattern x with weight vector in sm->w,
  i.e., computing argmax_{(y,h)} <w,psi(x,y,h)>. 
  Output pair (y,h) are stored at location pointed to by 
  pointers *y and *h. 
*/

		// 1. Write input to a file
		char *filename = "tmpfiles/classify_ex";
		// Online SVM: Note --- no longer using this code hence replace w by w_iters[0][0]
		write_to_file(x, y, sm->w_iters[0][0], sparm->max_feature_key, sparm->total_number_rels, filename);

		// 2. Call the ClassifyStructEgHelper method from JAVA
		system("export LD_LIBRARY_PATH=/usr/lib/lp_solve && java -cp java/bin:java/lib/* javaHelpers.ClassifyStructEgHelper tmpfiles/classify_ex");

		// 3. Read the values of *y and *h from the file tmpfiles/max_violator.result
		char *filename_res = "tmpfiles/classify_ex.result";
		FILE *fp = fopen(filename_res,"r");
		if (fp==NULL) {
			printf("Cannot open output file %s!\n", filename);
			exit(1);
		}

		int num_ylabels;
		fscanf(fp,"%d\n",&num_ylabels);
		y->num_relations = num_ylabels;

		int i;
		y->relations = (int*)malloc(sizeof(int)*num_ylabels);
		for(i = 0; i < num_ylabels; i ++){
			int ylabel;
			fscanf(fp, "%d\n",&ylabel);
			y->relations[i] = ylabel;
		}

		int num_hlabels;
		fscanf(fp, "%d\n", &num_hlabels);
		h->num_mentions = num_hlabels;

		if(num_hlabels != x.num_mentions){
			printf("Something is wrong .. no. of h labels  %d not matching no. of mentions %d ... exiting \n",num_hlabels, x.num_mentions);
			exit(1);
		}

		h->mention_labels = (int*)malloc(sizeof(int)*num_hlabels);
		for(i = 0; i < num_hlabels; i ++){
			int hlabel;
			fscanf(fp, "%d\n", &hlabel);
			h->mention_labels[i] = hlabel;
		}

		// cleanup
		fclose(fp);
}

void write_to_file(PATTERN x, LABEL y, double *w, long num_of_features, long total_number_rels, char *filename){
		FILE *fp = fopen(filename,"w");
		if (fp==NULL) {
			printf("Cannot open output file %s!\n", filename);
			exit(1);
		}

		// Write the mentions of PATTERN x
		fprintf(fp,"%d\n",x.num_mentions);
		int i;
		for(i = 0; i < x.num_mentions; i ++){
			WORD *mention_features = (WORD*) x.mention_features[i].words;
			int f_sz = atoi(x.mention_features[i].userdefined);

			int l;
			//printf("%d\t",f_sz);
			for(l = 0; l < f_sz; l ++){
				fprintf(fp, "%d:%.1f ",mention_features[l].wnum, mention_features[l].weight);
			}
			fprintf(fp, "\n");
		}

		// Write the LABEL y
		fprintf(fp,"%d\n",y.num_relations);
		for(i = 0; i < y.num_relations; i ++){
			fprintf(fp,"%d\n",y.relations[i]);
		}

		//Write the w. One line for each label.
		fprintf(fp,"%ld\n",total_number_rels);
		fprintf(fp,"%ld\n",num_of_features);
		int rel_id;
		for(rel_id = 0; rel_id <= total_number_rels; rel_id++){ // Include the nil label id
			int f_id;
			for(f_id = 1; f_id <= num_of_features; f_id ++){ // Feature ids start from 1 and go upto num_of_features .. hence this
				int key = (rel_id * num_of_features) + f_id;
				fprintf(fp, "%f ",w[key]);
			}
			fprintf(fp, "\n");
		}

		fclose(fp);
}

void write_to_file_params_t(double *w, long num_of_features, long total_number_rels, char *filename){

	FILE *fp = fopen(filename,"w");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename);
		exit(1);
	}

	//Write the w. One line for each label.
	fprintf(fp,"%ld\n",total_number_rels);
	fprintf(fp,"%ld\n",num_of_features);
	long rel_id;
	for(rel_id = 0; rel_id <= total_number_rels; rel_id++){ // Include the nil label id
		long f_id;
		for(f_id = 1; f_id <= num_of_features; f_id ++){ // Feature ids start from 1 and go upto num_of_features .. hence this
			long key = (rel_id * num_of_features) + f_id;
			fprintf(fp, "%.16g ",w[key]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);


}

void write_to_file_params_t_augmented(double *w, long num_of_features, long total_number_rels, char *filename){

	FILE *fp = fopen(filename,"w");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename);
		exit(1);
	}

	//Write the w. One line for each label.
	fprintf(fp,"%ld\n",total_number_rels);
	fprintf(fp,"%ld\n",num_of_features);
	long rel_id;
	for(rel_id = 0; rel_id <= total_number_rels; rel_id++){ // Include the nil label id
		long f_id;
		for(f_id = 1; f_id <= num_of_features; f_id ++){ // Feature ids start from 1 and go upto num_of_features .. hence this
			long key = (rel_id * num_of_features) + f_id;
			fprintf(fp, "%.16g ",w[key]); // IMPT: SHOULD NOT ADD 'wprev'. Should pass 'u' as it it to the max_violator function
		}
		fprintf(fp, "\n");
	}

	fclose(fp);


}

void find_most_violated_constraint_marginrescaling_all_online(LABEL *ybar_all, LATENT_VAR *hbar_all,
		STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int numEgs, char *tmpdir, char *trainfile,
		double frac_sim, char *dataset_stats_file, double rho_admm,	long isExhaustive,
		long isLPrelaxation, double Fweight, int datasetStartIdx, int chunkSz,
		int eid, int chunkid){

	// 1. Write input to a file
	char *filename = (char*) malloc(100);
	strcpy(filename, tmpdir);
	strcat(filename, "max_violator_all");

	write_to_file_params_t_augmented(sm->w_iters[eid][chunkid], sparm->max_feature_key, sparm->total_number_rels, filename);

	// 2. Call the FindMaxViolatorHelperAll method from JAVA
//	char * cmd = "export LD_LIBRARY_PATH=/usr/lib/lp_solve && "
//			" java -Xmx8G -cp java/bin:java/lib/* "
//			" -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio124/cplex/bin/x86-64_sles10_4.1/:/usr/lib/lp_solve "
//			" javaHelpers.FindMaxViolatorHelperAll "
//			" tmpfiles/max_violator_all dataset/reidel_trainSVM.data 0.9 ";

	char *cmd = malloc(1000);
	// ON MONASH ....
//        strcpy(cmd,"export LD_LIBRARY_PATH=~/lsvm_code/libs/lp_solve/:~/lsvm_code/libs/mosek.5/5/tools/platform/linux64x86/bin/ && "
//           " java -Xmx8G -cp java/bin:java/lib/* "
//	   " -Djava.library.path=../../libs/x86-64_sles10_4.1/:../../libs/lp_solve "
//	   " javaHelpers.FindMaxViolatorHelperAll ");

	// LOCAL PATH
	strcpy(cmd, //"export LD_LIBRARY_PATH=~/Research/software/lp_solve/ && "
			" java -Xmx2G -cp java/bin:java/lib/* "
			//" -Djava.library.path=~/Research/software/x86-64_sles10_4.1/:~/Research/software/lp_solve/ "
			" javaHelpers.FindMaxViolatorHelperAll ");
	strcat(cmd,filename);
	strcat(cmd, " ");
	strcat(cmd, trainfile);
	strcat(cmd, " ");
	char double_str[5]; sprintf(double_str,"%g", frac_sim);
	strcat(cmd, double_str);
	strcat(cmd, " ");
	strcat(cmd, dataset_stats_file);
	strcat(cmd, " ");
	char rho_str[5]; sprintf(rho_str,"%g", rho_admm);
	strcat(cmd, rho_str);
	strcat(cmd, " ");
	char isExhaustive_str[5]; sprintf(isExhaustive_str,"%ld", isExhaustive);
	strcat(cmd, isExhaustive_str);
	strcat(cmd, " ");
	char isLPrelaxation_str[5]; sprintf(isLPrelaxation_str,"%ld", isLPrelaxation);
	strcat(cmd, isLPrelaxation_str);
	strcat(cmd, " ");
	char Fweight_str[10]; sprintf(Fweight_str, "%g", Fweight);
	strcat(cmd, Fweight_str);
	strcat(cmd, " ");
	char datasetStartIdxStr[5]; sprintf(datasetStartIdxStr,"%d", datasetStartIdx);
	strcat(cmd, datasetStartIdxStr);
	strcat(cmd, " ");
	char chunkSzStr[5]; sprintf(chunkSzStr, "%d", numEgs);
	strcat(cmd, chunkSzStr);

	printf("Executing cmd (onlineSVM) : %s\n", cmd);fflush(stdout);
	system(cmd);


	// 3. Read the values of ybar_all and hbar_all from the file tmpfiles/max_violator_all.result
//	char *filename_res = "tmpfiles/max_violator_all.result";
	char *filename_res = (char*) malloc(100);
	strcpy(filename_res, tmpdir);
	strcat(filename_res, "max_violator_all.result");

	FILE *fp = fopen(filename_res,"r");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename_res);
		exit(1);
	}
	int j;
	for(j = 0; j < numEgs; j ++){

		int num_ylabels;
		fscanf(fp,"%d\n",&num_ylabels);
		ybar_all[j].num_relations = num_ylabels;

		int i;
		ybar_all[j].relations = (int*)malloc(sizeof(int)*num_ylabels);
		for(i = 0; i < num_ylabels; i ++){
			int ylabel;
			fscanf(fp, "%d\n",&ylabel);
			ybar_all[j].relations[i] = ylabel;
		}

		int num_hlabels;
		fscanf(fp, "%d\n", &num_hlabels);
		hbar_all[j].num_mentions = num_hlabels;

		hbar_all[j].mention_labels = (int*)malloc(sizeof(int)*num_hlabels);
		for(i = 0; i < num_hlabels; i ++){
			int hlabel;
			fscanf(fp, "%d\n", &hlabel);
			hbar_all[j].mention_labels[i] = hlabel;
		}
	}

	free(filename);
	free(filename_res);
	free(cmd);
	fclose(fp);
}


void find_most_violated_constraint_marginrescaling_all(LABEL *ybar_all, LATENT_VAR *hbar_all, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int numEgs, char *tmpdir, char *trainfile, double frac_sim, char *dataset_stats_file, double rho_admm, long isExhaustive, long isLPrelaxation, double Fweight, int datasetStartIdx, int chunkSz){

	// 1. Write input to a file
	char *filename = (char*) malloc(100);
	strcpy(filename, tmpdir);
	strcat(filename, "max_violator_all");

	// Online SVM: Note --- no longer using this code hence replace w by w_iters[0][0]
	write_to_file_params_t(sm->w_iters[0][0], sparm->max_feature_key, sparm->total_number_rels, filename);

	// 2. Call the FindMaxViolatorHelperAll method from JAVA
//	char * cmd = "export LD_LIBRARY_PATH=/usr/lib/lp_solve && "
//			" java -Xmx8G -cp java/bin:java/lib/* "
//			" -Djava.library.path=/opt/ibm/ILOG/CPLEX_Studio124/cplex/bin/x86-64_sles10_4.1/:/usr/lib/lp_solve "
//			" javaHelpers.FindMaxViolatorHelperAll "
//			" tmpfiles/max_violator_all dataset/reidel_trainSVM.data 0.9 ";

	char *cmd = malloc(1000);
	strcpy(cmd,"export LD_LIBRARY_PATH=/usr/lib/lp_solve && "
			" java -Xmx1G -cp java/bin:java/lib/* "
			" -Djava.library.path='/home/ajay/Research/software/lp_solve/:/home/ajay/Research/software/mosek.5/5/tools/platform/linux64x86/bin/:/home/ajay/Research/software/x86-64_sles10_4.1/' "
			" javaHelpers.FindMaxViolatorHelperAll ");
	strcat(cmd,filename);
	strcat(cmd, " ");
	strcat(cmd, trainfile);
	strcat(cmd, " ");
	char double_str[5]; sprintf(double_str,"%g", frac_sim);
	strcat(cmd, double_str);
	strcat(cmd, " ");
	strcat(cmd, dataset_stats_file);
	strcat(cmd, " ");
	char rho_str[5]; sprintf(rho_str,"%g", rho_admm);
	strcat(cmd, rho_str);
	strcat(cmd, " ");
	char isExhaustive_str[5]; sprintf(isExhaustive_str,"%ld", isExhaustive);
	strcat(cmd, isExhaustive_str);
	strcat(cmd, " ");
	char isLPrelaxation_str[5]; sprintf(isLPrelaxation_str,"%ld", isLPrelaxation);
	strcat(cmd, isLPrelaxation_str);
	strcat(cmd, " ");
	char Fweight_str[10]; sprintf(Fweight_str, "%g", Fweight);
	strcat(cmd, Fweight_str);
	strcat(cmd, " ");
	char datasetStartIdx_str[10]; sprintf(datasetStartIdx_str, "%ld", datasetStartIdx);
	strcat(cmd, datasetStartIdx_str);
	strcat(cmd, " ");
	char chunkSz_str[10]; sprintf(chunkSz_str, "%ld", chunkSz);
	strcat(cmd, chunkSz_str);
	strcat(cmd, " ");

	printf("Executing cmd : %s\n", cmd);fflush(stdout);
	system(cmd);


	// 3. Read the values of ybar_all and hbar_all from the file tmpfiles/max_violator_all.result
//	char *filename_res = "tmpfiles/max_violator_all.result";
	char *filename_res = (char*) malloc(100);
	strcpy(filename_res, tmpdir);
	strcat(filename_res, "max_violator_all.result");

	FILE *fp = fopen(filename_res,"r");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename_res);
		exit(1);
	}
	int j;
	for(j = 0; j < numEgs; j ++){

		int num_ylabels;
		fscanf(fp,"%d\n",&num_ylabels);
		ybar_all[j].num_relations = num_ylabels;

		int i;
		ybar_all[j].relations = (int*)malloc(sizeof(int)*num_ylabels);
		for(i = 0; i < num_ylabels; i ++){
			int ylabel;
			fscanf(fp, "%d\n",&ylabel);
			ybar_all[j].relations[i] = ylabel;
		}

		int num_hlabels;
		fscanf(fp, "%d\n", &num_hlabels);
		hbar_all[j].num_mentions = num_hlabels;

		hbar_all[j].mention_labels = (int*)malloc(sizeof(int)*num_hlabels);
		for(i = 0; i < num_hlabels; i ++){
			int hlabel;
			fscanf(fp, "%d\n", &hlabel);
			hbar_all[j].mention_labels[i] = hlabel;
		}
	}

	free(filename);
	free(filename_res);
	free(cmd);
	fclose(fp);
}

void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/

	// 1. Write input to a file
	char *filename = "tmpfiles/max_violator";
	// Online SVM: Note --- no longer using this code hence replace w by w_iters[0][0]
	write_to_file(x, y, sm->w_iters[0][0], sparm->max_feature_key, sparm->total_number_rels, filename);

	// 2. Call the FindMaxViolatorHelper method from JAVA
	system("export LD_LIBRARY_PATH=/usr/lib/lp_solve && java -Xmx1G -cp java/bin:java/lib/* javaHelpers.FindMaxViolatorHelper tmpfiles/max_violator");

	// 3. Read the values of ybar and hbar from the file tmpfiles/max_violator.result
	char *filename_res = "tmpfiles/max_violator.result";
	FILE *fp = fopen(filename_res,"r");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename);
		exit(1);
	}

	int num_ylabels;
	fscanf(fp,"%d\n",&num_ylabels);
	ybar->num_relations = num_ylabels;

	int i;
	ybar->relations = (int*)malloc(sizeof(int)*num_ylabels);
	for(i = 0; i < num_ylabels; i ++){
		int ylabel;
		fscanf(fp, "%d\n",&ylabel);
		ybar->relations[i] = ylabel;
	}

	int num_hlabels;
	fscanf(fp, "%d\n", &num_hlabels);
	hbar->num_mentions = num_hlabels;

	if(num_hlabels != x.num_mentions){
		printf("Something is wrong .. no. of h labels  %d not matching no. of mentions %d ... exiting \n",num_hlabels, x.num_mentions);
		exit(1);
	}

	hbar->mention_labels = (int*)malloc(sizeof(int)*num_hlabels);
	for(i = 0; i < num_hlabels; i ++){
		int hlabel;
		fscanf(fp, "%d\n", &hlabel);
		hbar->mention_labels[i] = hlabel;
	}

	fclose(fp);

}

void infer_latent_variables_all(LATENT_VAR *imputed_h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
		int numEgs, char* tmpdir, char *trainfile, int datasetStartIdx, int chunkSz, int eid, int chunkid){

	// 1. Write input to a file
	//char *filename = "tmpfiles/inf_lat_var_all";
	char *filename = (char*) malloc(100);
	strcpy(filename, tmpdir);
	strcat(filename,"inf_lat_var_all");

	printf("(onlinesvm) Before writing params to file "); fflush(stdout);
	write_to_file_params_t(sm->w_iters[eid][chunkid], sparm->max_feature_key, sparm->total_number_rels, filename);
	printf("(onlinesvm) After writing params to file "); fflush(stdout);

	// 2. Call the InferLatentVarHelperAll method from JAVA
	// TODO: Modifiy the cmds appropriately
//	char * cmd = " export LD_LIBRARY_PATH=/usr/lib/lp_solve && "
//			" java -Xmx8G -cp java/bin:java/lib/* javaHelpers.InferLatentVarHelperAll "
//			" tmpfiles/inf_lat_var_all dataset/reidel_trainSVM.data ";
	char *cmd = malloc(1000);
	strcpy(cmd,"export LD_LIBRARY_PATH=/usr/lib/lp_solve && "
			"java -Xmx8G -cp java/bin:java/lib/*  "
			"-Djava.library.path='/home/ajay/Research/software/lp_solve/:/home/ajay/Research/software/mosek.5/5/tools/platform/linux64x86/bin/:/home/ajay/Research/software/x86-64_sles10_4.1/' "
			"javaHelpers.InferLatentVarHelperAll ");
	strcat(cmd,filename);
	//strcat(command," dataset/reidel_trainSVM.data");
	strcat(cmd, " ");
	strcat(cmd, trainfile);
	strcat(cmd, " ");
	char datasetStartIdx_str[10]; sprintf(datasetStartIdx_str, "%ld", datasetStartIdx);
	strcat(cmd, datasetStartIdx_str);
	strcat(cmd, " ");
	char chunkSz_str[10]; sprintf(chunkSz_str, "%ld", chunkSz);
	strcat(cmd, chunkSz_str);
	strcat(cmd, " ");

		//printf("Running : %s\n", command);
	printf("Executing cmd (online SVM): %s\n", cmd);fflush(stdout);
	system(cmd);

	// 3. Read the values of ybar_all and hbar_all from the file tmpfiles/max_violator_all.result
//	char *filename_res = "tmpfiles/inf_lat_var_all.result";
	char *filename_res = (char*) malloc(100);
	strcpy(filename_res, tmpdir);
	strcat(filename_res, "inf_lat_var_all.result");

	FILE *fp = fopen(filename_res,"r");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename_res);
		exit(1);
	}
	int j;
	for(j = 0; j < numEgs; j ++){
		int num_mentions;
		fscanf(fp,"%d\n",&num_mentions);

		imputed_h[j].num_mentions = num_mentions;
		imputed_h[j].mention_labels = (int*)malloc(sizeof(int)*num_mentions);
		int i;
		for(i = 0; i < num_mentions; i ++){
			int label;
			fscanf(fp, "%d\n", &label);
			imputed_h[j].mention_labels[i] = label;
		}
	}

	free(cmd);
	free(filename);
	free(filename_res);
	fclose(fp);

}

LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/

  LATENT_VAR h;

  // 1. Write input to a file
  char *filename = "tmpfiles/inf_lat_var";
  // Online SVM: Note --- no longer using this code hence replace w by w_iters[0][0]
  write_to_file(x, y, sm->w_iters[0][0], sparm->max_feature_key, sparm->total_number_rels, filename);

  // 2. CALL the method InferLatentVarHelper FROM JAVA
  //printf("export LD_LIBRARY_PATH=/usr/lib/lp_solve && java -cp java/bin:java/lib/* javaHelpers.InferLatentVarHelper tmpfiles/inf_lat_var \n");
  system("export LD_LIBRARY_PATH=/usr/lib/lp_solve && java -cp java/bin:java/lib/* javaHelpers.InferLatentVarHelper tmpfiles/inf_lat_var");

  // 3. READ the LATENT VAR h from the file tmpfiles/inf_lat_var.latentvar
  FILE *fp = fopen("tmpfiles/inf_lat_var.latentvar","r");
  if (fp==NULL) {
	  printf("Cannot open output file %s!\n", "tmpfiles/inf_lat_var");
	  exit(1);
  }
  int num_mentions;
  fscanf(fp,"%d\n",&num_mentions);

  if(num_mentions != x.num_mentions){
	  printf("Something is wrong .. exiting ...");
	  exit(1);
  }

  h.num_mentions = num_mentions;
  h.mention_labels = (int*)malloc(sizeof(int)*num_mentions);
  int i;
  for(i = 0; i < num_mentions; i ++){
	  int label;
	  fscanf(fp, "%d\n", &label);
	  h.mention_labels[i] = label;
  }

  // CLEANUP
  fclose(fp);
  // Delete tmp files created.

  return(h); 
}

int is_present(int ybar_label, LABEL y){
	int i;
	for(i = 0; i < y.num_relations; i++)
		if(ybar_label == y.relations[i])
			return 1;

	return 0;
}

void create_vector(LABEL y, int *y_vec, LABEL ybar, int *ybar_vec) {

	int i;
	if(y.num_relations == 0)
		y_vec[0] = 1;
	else {
		for(i = 0; i < y.num_relations; i ++){
			y_vec[y.relations[i]] = 1;
		}
	}
	if(ybar.num_relations == 0)
		ybar_vec[0] = 1;
	else {
		for(i = 0; i < ybar.num_relations; i ++){
			ybar_vec[ybar.relations[i]] = 1;
		}
	}
}

// 08/07/2014 : loss function based on F1
// 03/09/2014 : Modified the loss to take weights for differently weigting P and R
//				Fw = 1 / [w/P + (1-w)/R]. where w (0,1) For F1, w = 0.5
// In the loss-aug inference, dual decomposition, only the mesh creation (offline) shld take this new formula
// Other aspects remain unchanged.
double lossF1(EXAMPLE *ex, int numEgs, LABEL *ybar_all, STRUCT_LEARN_PARM *sparm, double Fweight) {
	double loss = 0.0;

	int num_non_nil_rels = sparm->total_number_rels;

	int i;
	int FP = 0, FN = 0, Np = 0;

	for(i = 0; i < numEgs; i ++){
		LABEL y_i = ex[i].y;
		LABEL ytilde_i = ybar_all[i];

		int * y_i_vec = (int*) calloc(num_non_nil_rels+1, sizeof(int));
		int * ytilde_i_vec = (int*) calloc(num_non_nil_rels+1, sizeof(int));

		create_vector(y_i, y_i_vec, ytilde_i, ytilde_i_vec);

		int l;
		for(l = 1; l <= num_non_nil_rels; l ++){
			FP += ytilde_i_vec[l] * (1 - y_i_vec[l]);
			FN += (1 - ytilde_i_vec[l]) * y_i_vec[l];
			Np += y_i_vec[l];
		}

		free(y_i_vec);
		free(ytilde_i_vec);
	}

	// loss = (double)(FP + FN) / (2*Np + FP - FN);

	loss = (double)( (Fweight * FP) + ((1 - Fweight) * FN) ) / (Np + ( Fweight * (FP - FN) ) );

	//printf("Loss (F1) is %f\n",loss);

	return (loss);
}

// 11/05/2014 : New loss function -- correct implementation of hamming loss for our problem
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
	double ans = 0;

	int num_non_nil_rels = sparm->total_number_rels;

	int * y_vec = (int*) calloc(num_non_nil_rels+1, sizeof(int));
	int * ybar_vec = (int*) calloc(num_non_nil_rels+1, sizeof(int));

	int i;
	if(y.num_relations == 0)
		y_vec[0] = 1;
	else {
		for(i = 0; i < y.num_relations; i ++){
			y_vec[y.relations[i]] = 1;
		}
	}
	if(ybar.num_relations == 0)
		ybar_vec[0] = 1;
	else {
		for(i = 0; i < ybar.num_relations; i ++){
			ybar_vec[ybar.relations[i]] = 1;
		}
	}

	for(i = 0; i <= num_non_nil_rels; i ++){
		if(ybar_vec[i] != y_vec[i])
			ans += 1.0;
	}

	return(ans);
}

// Old loss function : not correct version of hamming loss
//double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
///*
//  Computes the loss of prediction (ybar,hbar) against the
//  correct label y.
//*/
//  double ans = 0;
//
//  // Loss is defined as the num of incorrect labels in ybar
//  // (in comparison with gold std. y)
//  int i;
//  for(i = 0; i < ybar.num_relations; i ++){
//	  int ybar_label = ybar.relations[i];
//	  if(!is_present(ybar_label,y))
//		  ans += 1.0;
//  }
//
//  return(ans);
//}

void write_to_file_params_t_online(double ***w_iters, int totalEpochs, int numChunks, long num_of_features, long total_number_rels, char *filename){

	FILE *fp = fopen(filename,"w");
	if (fp==NULL) {
		printf("Cannot open output file %s!\n", filename);
		exit(1);
	}

	fprintf(fp,"%d\n", totalEpochs);
	fprintf(fp,"%d\n", numChunks);

	//Write the w. One line for each label.
	fprintf(fp,"%ld\n",total_number_rels);
	fprintf(fp,"%ld\n",num_of_features);
	long rel_id;
	int eid, chunkid;
	fprintf(fp, "==\n");
	for(eid = 0; eid < totalEpochs; eid++){
		for(chunkid = 0; chunkid < numChunks; chunkid++){

			double *w = w_iters[eid][chunkid];

			for(rel_id = 0; rel_id <= total_number_rels; rel_id++){ // Include the nil label id
				long f_id;
				for(f_id = 1; f_id <= num_of_features; f_id ++){ // Feature ids start from 1 and go upto num_of_features .. hence this
					long key = (rel_id * num_of_features) + f_id;
					fprintf(fp, "%.16g ",w[key]);
				}
				fprintf(fp, "\n");
			}
			fprintf(fp, "--\n");
		}

		fprintf(fp, "==\n");
	}

	fclose(fp);


}

void write_struct_model_online(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, int totalEpochs, int numChunks) {
/*
  Writes the learned weight vector sm->w_iters to file after training.
*/
	write_to_file_params_t_online(sm->w_iters, totalEpochs, numChunks, sparm->max_feature_key, sparm->total_number_rels, file);
}

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Writes the learned weight vector sm->w to file after training. 
*/
/*	  FILE *modelfl;
	  long i;

	  modelfl = fopen(file,"w");
	  if (modelfl==NULL) {
	    printf("Cannot open model file %s for output!", file);
		exit(1);
	  }

	  fprintf(modelfl, "# sizePsi:%ld\n", sm->sizePsi);
	  for (i=1;i<sm->sizePsi+1;i++) {
	    fprintf(modelfl, "%ld:%.16g\n", i, sm->w[i]);
	  }
	  fclose(modelfl);
 */
	// Online SVM: Note --- no longer using this code hence replace w by w_iters[0][0]
	write_to_file_params_t(sm->w_iters[0][0], sparm->max_feature_key, sparm->total_number_rels, file);
}

//STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {
///*
//  Reads in the learned model parameters from file into STRUCTMODEL sm.
//  The input file format has to agree with the format in write_struct_model().
//*/
//
//	STRUCTMODEL sm;
//	FILE *modelfl;
//	long sizePsi, i, fnum;
//	double fweight;
//
//	modelfl = fopen(file,"r");
//	if (modelfl==NULL) {
//		printf("Cannot open model file %s for input!", file);
//		exit(1);
//	}
//
//	if (fscanf(modelfl, "# sizePsi:%ld", &sizePsi)!=1) {
//		printf("Incorrect model file format for %s!\n", file);
//		fflush(stdout);
//	}
//
//	sm.sizePsi = sizePsi;
//	sm.w = (double*)malloc(sizeof(double)*(sizePsi+1));
//	for (i=0;i<sizePsi+1;i++) {
//		sm.w[i] = 0.0;
//	}
//
//	while (!feof(modelfl)) {
//		fscanf(modelfl, "%ld:%lf", &fnum, &fweight);
//		sm.w[fnum] = fweight;
//	}
//	fclose(modelfl);
//
//	return(sm);
//
//}

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
	// OnlineSVM .. no longer used .. so using w_iters[0][0] arbitrarily
	  free(sm.w_iters[0][0]);
}

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x. 
*/
	// TODO: Ajay: hopefully this is correct. Else check at the end
	int i;
	for(i = 0; i < x.num_mentions; i ++){
		free(x.mention_features[i].words); // Free each of the WORD* vector
	}
	free(x.mention_features); // Free the SVECTOR* vector finally.
}

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
*/
	// TODO: Ajay: hopefully this is correct. Else check at the end
	free(y.relations);
} 

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
*/
	// TODO: Ajay: hopefully this is correct. Else check at the end
	free(h.mention_labels);
}

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/
  int i;
  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
  
  /* set default */
  
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}

