/* 21 December 2008 */
// modified the interface for loss()
/* 7 October 2008 */
#include "./svm_light/svm_common.h"
#include "svm_struct_latent_api_types.h"

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm);
void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm);
void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
LATENT_VAR infer_latent_variables(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
//double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);
void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm);
void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm);
void free_pattern(PATTERN x);
void free_label(LABEL y);
void free_latent_var(LATENT_VAR h);
void free_struct_sample(SAMPLE s);
void parse_struct_parameters(STRUCT_LEARN_PARM *sparm);
void print_motif(STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);

double pairwise_loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);
double mitre_loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm);






