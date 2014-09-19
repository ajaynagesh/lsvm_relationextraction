package javaHelpers;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;
import javaHelpers.Utils.Region;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

public class CplexHelper {
	
	static ArrayList<IloNumVar[]> initVariablesYtilde(IloCplex model, int datasetsize, int numPosLabels) throws IloException{
		
		ArrayList<IloNumVar[]> variables = new ArrayList<IloNumVar[]>();
		
		for(int i = 0; i < datasetsize; i ++){
			
			// float variables between 0 and 1. Basically denoting that it is LP
			IloNumVar[] vars_i = model.numVarArray(numPosLabels, 0, 1);
			
			// int variables
			//IloNumVar[] vars_i = model.intVarArray(numPosLabels, 0, 1);
			
			variables.add(vars_i);
		}
		
		return variables;
		
	}
	
	// TODO: Create the hidden variables (h)
	static ArrayList<ArrayList<IloNumVar[]>> initVariableLatent(IloCplex model, ArrayList<DataItem> dataset, int numPosLabels) throws IloException{

		ArrayList<ArrayList<IloNumVar[]>> allLatentVars = new ArrayList<ArrayList<IloNumVar[]>>();
		
		for(int i = 0; i < dataset.size(); i++){
			DataItem dataPt_i = dataset.get(i);
			int numOfMentions_i = dataPt_i.pattern.size();
			
			ArrayList<IloNumVar[]> latentvars_i  = new ArrayList<IloNumVar[]>();
			for(int m = 0; m < numOfMentions_i; m ++){
				// float variables between 0 and 1. Basically denoting that it is LP
				IloNumVar[] h_mention = model.numVarArray(numPosLabels+1, 0, 1);  // NOTE: Also include the nil label (indexed by 0)
				
				// int variables between 0 and 1. Basically denoting that it is LP
				//IloNumVar[] h_mention = model.intVarArray(numPosLabels+1, 0, 1);  // NOTE: Also include the nil label (indexed by 0)
				
				latentvars_i.add(h_mention);
			}
			allLatentVars.add(latentvars_i);
		}
		
		return allLatentVars;
		
	}
	
	public static Pair<Double, Double> computeConstraintCoeff(double[] A, double[] B, double[] C, int y_il ){
		
		
		double k1 = ( (B[0] - A[0])*(C[1] - A[1]) ) - ( (B[1] - A[1])*(C[0] - A[0]) ); // (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
		double k2 = ( B[0] - A[0] ); // Bx - Ax
		double k3 = ( B[1] - A[1] ); // By - Ay
		
		double coeff = -k1*k2*y_il - k1*k3*(1 - y_il); // -k1*k2*y_i,l - k1*k3*(1-y_i,l) 
				
		double constant = k1 * k2 * y_il; //Constant in the constraint
		
		return new Pair<Double, Double>(coeff, constant);
		
	}
	
	public static int[] initVec(int ygold[], int sz){
		int yi[] = new int[sz+1]; // 0 position is nil label and is not filled; so one extra element is  created ( 1 .. 51)
		Arrays.fill(yi, 0);
		
		for(int y : ygold)
			yi[y] = 1;

		return yi;
	}
	
	static double buildCplexModel(IloCplex cplexModel, ArrayList<IloNumVar[]> vars_ytilde, Region r,
			ArrayList<DataItem> dataset, int numPosLabels,
			ArrayList<ArrayList<IloNumVar[]>> vars_latent, LabelWeights [] zWeights) throws IloException{
		
		double A[] = r.p1;
		double B[] = r.p2;
		double C[] = r.p3;
		double alpha_r = r.coeff[0];
		double beta_r = r.coeff[1];
		double gamma_r = r.coeff[2];

		double regionConstant = gamma_r;
		
		IloLinearNumExpr objective = cplexModel.linearNumExpr();
		IloLinearNumExpr constraint1 = cplexModel.linearNumExpr();
		double cons_constraint1 = 0.0;
		IloLinearNumExpr constraint2 = cplexModel.linearNumExpr();
		double cons_constraint2 = 0.0;
		IloLinearNumExpr constraint3 = cplexModel.linearNumExpr();
		double cons_constraint3 = 0.0;

		for(int i = 0; i < dataset.size(); i ++){ // For every datum in training dataset

			
			///*******************************************************************************************************************************
			/// ********************************************CPLEX MODEL FOR  LOSS <START> **************************************************** 
			int yi[] = initVec(dataset.get(i).ylabel, numPosLabels);
			
			IloNumVar[] vars_ytilde_i = vars_ytilde.get(i);
			
			for(int l = 1; l <= numPosLabels; l ++){ // For every label-id

				////////////////////////////// ~y_i,l //////////////////////////////////////////////////

				double coeff = (alpha_r * (1 - yi[l])) 	// alpha_r * (1 - y_i,l)
								- (beta_r * yi[l]);			// beta_r * y_i,l

				// Add ~y_i,l to the objective function
				IloNumVar var = vars_ytilde_i[l-1]; // NOTE .. l-1 .. for ~y_i,l
				objective.addTerm(coeff, var);

				regionConstant += beta_r *  yi[l];					// \sum_i,l beta_r * y_i,l

				// Add ~y_i,l to the first constraint
				Pair<Double, Double> c1_coeffs = computeConstraintCoeff(A, B, C, yi[l]);
				constraint1.addTerm(c1_coeffs.first(), var);
				cons_constraint1 += c1_coeffs.second();

				// Add ~y_i,l to the second constraint
				Pair<Double, Double> c2_coeffs = computeConstraintCoeff(B, C, A, yi[l]);
				constraint2.addTerm(c2_coeffs.first(), var);
				cons_constraint2 += c2_coeffs.second();

				// Add ~y_i,l to the third constraint
				Pair<Double, Double> c3_coeffs = computeConstraintCoeff(A, C, B, yi[l]);
				constraint3.addTerm(c3_coeffs.first(), var);
				cons_constraint3 += c3_coeffs.second();

				////////////////////////////// ~y_i,l //////////////////////////////////////////////////

			} // END: For every label-id

			///*******************************************************************************************************************************
			/// ********************************************CPLEX MODEL FOR  LOSS <END> ******************************************************
			
			/// ---------------------------------------------------------------------------------------------------------------------------------------------
			/// ---------------------------------------------------------------------------------------------------------------------------------------------
			
			///*******************************************************************************************************************************
			/// ********************************************CPLEX MODEL FOR  MARKOV RANDOM FIELD <START> ******************************************************
			
			DataItem entityPair_i = dataset.get(i);
			int [] yLabelsGold_i = entityPair_i.ylabel;
			int numMentions_i = entityPair_i.pattern.size();
			ArrayList<Counter<Integer>> x_i = entityPair_i.pattern; 
			List<Counter<Integer>> scores_i = Utils.computeScores(x_i, zWeights, yLabelsGold_i);
			ArrayList<IloNumVar[]> vars_h_i = vars_latent.get(i);
			
			/// (Model) MRF Objective for i ... MO_i 
			for(int m = 0; m < numMentions_i; m++){
				for(int l = 0; l <= numPosLabels; l++){ // NOTE: For the latent variables, we need to iterate over the NIL label ... hence .. 0 to numPosLabels+1 
					IloNumVar var =  vars_h_i.get(m)[l];
					double coeff = scores_i.get(m).getCount(l);
					objective.addTerm(coeff, var);
				}
			}
			/// (Model) MRF Objective for i ... MO_i <end>
			
			/// MC^(1)_i
			for(int m = 0; m < numMentions_i; m++){
				IloLinearNumExpr cons_type1 = cplexModel.linearNumExpr();
				for(int l = 0; l <= numPosLabels; l ++){ // NOTE: include the NIL label for the latent variables constraints
					cons_type1.addTerm(1, vars_h_i.get(m)[l]);
				}
				cplexModel.addEq(cons_type1, 1);
			}
			/// MC^(1)_i <end>
			
			/// MC^(2)_i
			for(int m = 0; m < numMentions_i; m++){
				for(int l = 1; l <= numPosLabels; l ++){ // NOTE: do not include NIL label for the  case involving ~y_i,l variables
					IloLinearNumExpr cons_type2 = cplexModel.linearNumExpr();
					cons_type2.addTerm(1, vars_h_i.get(m)[l]);
					cons_type2.addTerm(-1, vars_ytilde_i[l-1]); // NOTE the l-1 for ~y_i,l
					cplexModel.addLe(cons_type2, 0);
				}
			}
			/// MC^(2)_i <end>
			
			/// MC^(3)_i
			for(int l = 1; l <= numPosLabels; l++){ // NOTE: do not include the NIL label for  the case involving ~y,i,l variables
				IloLinearNumExpr cons_type3 = cplexModel.linearNumExpr();
				for(int m = 0; m < numMentions_i; m ++){
					cons_type3.addTerm(1, vars_h_i.get(m)[l]);
				}
				cons_type3.addTerm(-1, vars_ytilde_i[l-1]); // NOTE the l-1 for ~y_i,l
				cplexModel.addGe(cons_type3, 0);
			}
			/// MC^(3)_i <end>
			
			
			///*******************************************************************************************************************************
			/// ********************************************CPLEX MODEL FOR  MARKOV RANDOM FIELD <END> ******************************************************
					
			
		} // END: For every datum in training dataset

		Triple<Double, Double, Double> c1 =  compute_k1k2k3 (A, B, C);
		cons_constraint1 += ((-c1.first() * c1.second() * A[1]) + (c1.first() * c1.third() * A[0]));

		Triple<Double, Double, Double> c2 =  compute_k1k2k3 (B, C, A);
		cons_constraint2 += ((-c2.first() * c2.second() * B[1]) + (c2.first() * c2.third() * B[0]));

		Triple<Double, Double, Double> c3 =  compute_k1k2k3 (A, C, B);
		cons_constraint3 += ((-c3.first() * c3.second() * A[1]) + (c3.first() * c3.third() * A[0]));
		
		// Add the 3 region constraints to the LP problem.
		cplexModel.addGe(constraint1, -cons_constraint1);
		cplexModel.addGe(constraint2, -cons_constraint2);
		cplexModel.addGe(constraint3, -cons_constraint3);

		// Create the LP/ILP problem object and add the objective (constructed from LossOpt and Model[MRF]Opt)  
		cplexModel.addMaximize(objective);
				
		return regionConstant;
	}
	
	public static Triple<Double, Double, Double> compute_k1k2k3(double[] A, double[] B, double[] C){
		
		double k1 = ( (B[0] - A[0])*(C[1] - A[1]) ) - ( (B[1] - A[1])*(C[0] - A[0]) ); // (Bx - Ax)(Cy - Ay) - (By - Ay)(Cx - Ax)
		double k2 = ( B[0] - A[0] ); // Bx - Ax
		double k3 = ( B[1] - A[1] ); // By - Ay
		
		return new Triple<Double, Double, Double>(k1, k2, k3);
	}

}
