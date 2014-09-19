package javaHelpers;

import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.concurrent.Callable;

import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;
import javaHelpers.Utils.Region;

public class OptimizeLossAugForRegion implements Callable<Double>{

	Region r;
	ArrayList<DataItem> dataset;
	LabelWeights [] zWeights;
	int numPosLabels;
	
	public OptimizeLossAugForRegion(Region region, ArrayList<DataItem> dataset, int numPosLabels, LabelWeights[] zWeights ){
		this.r = region;
		this.dataset = dataset;
		this.numPosLabels = numPosLabels;
		this.zWeights = zWeights;
	}
	
	@Override
	public Double call() throws Exception {
		
		long startilpsol = System.currentTimeMillis();

		// Build the LP problem model
		IloCplex cplexModel;
		cplexModel = new IloCplex();

		// NOTE: both the set of variables Ytilde and H are created as real variables taking values ... (0,1)
		ArrayList<IloNumVar[]> vars_Ytilde = CplexHelper.initVariablesYtilde(cplexModel,  dataset.size(), numPosLabels); 
		ArrayList<ArrayList<IloNumVar[]>> vars_latent = CplexHelper.initVariableLatent(cplexModel, dataset, numPosLabels);
		
		// NOTE: return value is constant for a region in the LP formula. Added to the solved value of max objective
		double regionObjValue = CplexHelper.buildCplexModel(cplexModel, vars_Ytilde, r, dataset, numPosLabels, vars_latent, zWeights); 
		
		// Solve the LP
		if ( cplexModel.solve() ) {

			regionObjValue += cplexModel.getObjValue();

			System.out.println("Solution status = " + cplexModel.getStatus());
			System.out.println(" cost = " + regionObjValue);
		}
		
		cplexModel.end();
		long endilpsol = System.currentTimeMillis();
		double time = (double)(endilpsol - startilpsol) / 1000.0;
		System.out.println("Log: Total time for the LP problem (in thread): " + time + " s");

		return regionObjValue;
	}

}
