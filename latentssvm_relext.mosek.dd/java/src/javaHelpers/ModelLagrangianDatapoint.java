package javaHelpers;

import ilpInference.InferenceWrappers;
import ilpInference.YZPredicted;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Callable;
import javaHelpers.FindMaxViolatorHelperAll.LabelWeights;

import edu.stanford.nlp.stats.Counter;

public class ModelLagrangianDatapoint implements Callable<YZPredicted>{

	DataItem example;
	LabelWeights [] zWeights;
	double [] lambda_i;
	
	public ModelLagrangianDatapoint(DataItem example,LabelWeights [] zWeights, double [] lambda_i){
		this.example = example;
		this.zWeights = zWeights;
		this.lambda_i = lambda_i;	
	}
	
	@Override
	public YZPredicted call() throws Exception {
		// TODO Auto-generated method stub
		
		int [] yLabelsGold = example.ylabel;
		int numMentions = example.pattern.size();
		ArrayList<Counter<Integer>> pattern = example.pattern; 

		List<Counter<Integer>> scores = Utils.computeScores(pattern, zWeights, yLabelsGold);

		Set<Integer> yLabelsSetGold = new HashSet<Integer>();
		for(int y : yLabelsGold)  
			yLabelsSetGold.add(y);

		InferenceWrappers ilp = new InferenceWrappers();
		//YZPredicted yz = ilp.generateYZPredictedILP_loss(scores, numMentions, 0, yLabelsGold);
		YZPredicted yz = ilp.generateYZPredictedILP_lagrangian(scores, numMentions, 0, yLabelsGold, lambda_i);
		//YtildeDashStar.add(yz);
		return yz;
	}

}
