Code for Latent Structural SVM (June 4 2009)
--------------------------------------------
(implemented by Chun-Nam Yu based on the SVM-light code by Thorsten Joachims,
with the Mosek optimization software. Standalone version coming soon.)


INSTALLATION
------------
1. This software uses the Mosek quadratic program solver [2]. Download and install Mosek from www.mosek.com following the instructions on their website (they've provided a evaluation license and also a special free license for students)
2. After Mosek has been installed, modify the Makefile in latentssvm by replacing 'your_mosek_directory' with the path where you install Mosek.
3. Type 'make' in the directory latentssvm and make sure that the compilation is successful.


USING THE LATENT SVM^STRUCT API
-------------------------------
1. This implemention of latent structural SVM follows the modular design of SVM^struct [3]. To implement your own latent structural SVM application, you need to modify the two files svm_struct_latent_api_types.h and svm_struct_latent_api.c. 
2. An example implementation on the application of motif finding in [1] is available for download at http://www.cs.cornell.edu/~cnyu/latentssvm. 
3. You may also find the set of example applications on the SVM^struct website instructive when implementing your own latent structural SVM applications. The structural SVM application examples are available at http://svmlight.joachims.org/svm_struct.html.


CONTACT
-------
If you had any suggestions to the program or have bugs to report, you can email Chun-Nam Yu at cnyu@cs.cornell.edu.  


REFERENCES
----------
[1] C.-N. Yu and T. Joachims: Learning Structural SVMs with Latent Variables, ICML 2009 
[2] The Mosek Optimization Software [www.mosek.com]
[3] I. Tsochantaridis, T. Hofmann, T. Joachims, and Y. Altun: Support Vector Learning for Interdependent and Structured Output Spaces, ICML 2004
