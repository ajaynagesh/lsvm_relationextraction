package ilpInference;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import net.sf.javailp.Constraint;
import net.sf.javailp.Linear;
import net.sf.javailp.OptType;
import net.sf.javailp.Problem;
import net.sf.javailp.Result;
import net.sf.javailp.Solver;
import net.sf.javailp.SolverFactory;
import net.sf.javailp.SolverFactoryLpSolve;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ErasureUtils;
import edu.stanford.nlp.util.Index;

public class InferenceWrappers {
	
	public int [] generateZUpdateILP(List<Counter<Integer>> scoresYGiven, 
											  int numOfMentions, 
											  Set<Integer> goldPos,
											  int nilIndex){
//		System.out.println("Calling ILP inference for Pr (Z | Y,X)");
//		System.out.println("Num of mentions : " + numOfMentions);
//		System.out.println("Relation labels : " + goldPos);

		int [] zUpdate = new int[numOfMentions];
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		Linear objective = new Linear();
		Linear constraint;

		if(goldPos.size() > numOfMentions){
			//////////////Objective --------------------------------------
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				Counter<Integer> score = scoresYGiven.get(mentionIdx);
				for(int label : score.keySet()){
					if(label == nilIndex)
						continue; 
					
					String var = "z"+mentionIdx+"_"+"y"+label;
					double coeff = score.getCount(label);
					objective.add(coeff, var);

					//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
				}
			}
		
			problem.setObjective(objective, OptType.MAX);
			
			/////////// Constraints ------------------------------------------
			
			/// 1. \Sum_{i \in Y'} z_ji = 1 \forall j
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				constraint = new Linear();
				for(int y : goldPos){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
						
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
				}
								
				problem.add(constraint, "=", 1);
				//System.out.println(" 0 = "+ "1");
			}
			
			/// 2. \Sum_j z_ji <= 1 \forall i
			for(int y : goldPos){
				constraint = new Linear();
				for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
				}
				problem.add(constraint, "<=", 1);
			}
		}	
		else {
			//////////////Objective --------------------------------------
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				Counter<Integer> score = scoresYGiven.get(mentionIdx);
				for(int label : score.keySet()){
					String var = "z"+mentionIdx+"_"+"y"+label;
					double coeff = score.getCount(label);
					objective.add(coeff, var);

					//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
				}
			}

			problem.setObjective(objective, OptType.MAX);

			/// 1. equality constraints \Sum_{i \in Y'} z_ji = 1 \forall j
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				constraint = new Linear();
				for(int y : goldPos){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
						
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
				}
				constraint.add(1, "z"+mentionIdx+"_"+"y"+nilIndex); //nil index added to constraint
				
				problem.add(constraint, "=", 1);
				//System.out.println(" 0 = "+ "1");
			}
			
			//System.out.println("\n-----------------");
			/// 2. inequality constraint ===>  1 <= \Sum_j z_ji \forall i \in Y'  {lhs=1, since we consider only Y' i.e goldPos}
			/////////// ------------------------------------------------------
			for(int y : goldPos){
				constraint = new Linear();
				for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
					String var = "z"+mentionIdx+"_"+"y"+y;
					constraint.add(1, var);
					//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
				}
				problem.add(constraint, ">=", 1);
				//System.out.println(" 0 - " + "y"+y +" >= 0" );
			}
			/////////// ------------------------------------------------------
		}
				
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		//System.out.println("Result : " + result);
		
		if(result == null){
			System.out.println("Num of variables : " + problem.getVariablesCount());
			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
			System.out.println("Objective Function : ");
			System.out.println(problem.getObjective());
			System.out.println("Constraints : ");
			for(Constraint c : problem.getConstraints())
				System.out.println(c);
			
			System.out.println("Result is NULL ... Error ...");
			
			System.exit(0);

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				String [] split = var.toString().split("_");
				//System.out.println(split[0]);
				int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
				//System.out.println(split[1]);
				int ylabel = Integer.parseInt(split[1].toString().substring(1));
				if(ylabel != nilIndex)
					zUpdate[mentionIdx] = ylabel;
			}			
		}
	
		return zUpdate;
	}
	
	public YZPredicted generateYZPredictedILPnoisyOr(List<Counter<Integer>> scores,
			  int numOfMentions, 
			  Index<String> yLabelIndex, 
			  Counter<Integer> typeBiasScores,
			  int egId,
			  int epoch,
			  int nilIndex){
		
		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		Linear objective = new Linear();
	
		//////////////Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);

				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		for(String label : yLabelIndex){
			int y = yLabelIndex.indexOf(label);
			String var = "e"+y;
			objective.add(-1, var);
			
		}
				
		problem.setObjective(objective, OptType.MAX);
		
		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			constraint = new Linear();
			for(String yLabel : yLabelIndex){
				int y = yLabelIndex.indexOf(yLabel);
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);

				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}

			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			for(String yLabel : yLabelIndex){
				constraint = new Linear();
				int y = yLabelIndex.indexOf(yLabel);
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		for(String yLabel : yLabelIndex){
			constraint = new Linear();
			int y = yLabelIndex.indexOf(yLabel);
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			constraint.add(1, "e"+y);
			problem.add(constraint, ">=", 0);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}

		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);

		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);

		if(result == null){
			System.out.println("Result is NULL ... Error in iter = " + epoch + " Eg Id : " + egId + " ...  Skipping this");
			return predictedVals;
		}

		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}

		
		return predictedVals;
	}
	
	boolean isPresent(int label, int[] goldlabels){
		
		boolean present = false;
		
		for(int goldy : goldlabels){
			if(label == goldy)
				present = true;
		}

		return present;
	}
		
	public YZPredicted generateYZPredictedILP_loss(List<Counter<Integer>> scores,
			  int numOfMentions,
			  int nilIndex,
			  int [] yLabelsGold){

		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		
		Linear objective = new Linear();
		
		////////////// Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);
				
				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		
		for(int label : scores.get(0).keySet()){
			// Simple loss function (not hamming loss)
			String var = "y" + label;
			double loss = 0.0;
			
//			if(label == nilIndex){
//				if(yLabelsGold.length == 0){
//					objective.add(1.0, var);
//				}
//			}
//			else {
//				if(!isPresent(label, yLabelsGold)){
//					objective.add(1.0, var);
//				}
//			}
			
			if(!isPresent(label, yLabelsGold))
				loss = 1.0;
			
			objective.add(loss, var);
		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		 
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			constraint = new Linear();
			for(int y : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int y : score.keySet()){
				constraint = new Linear();
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		//System.out.println("\n-----------------");
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		
		for(int y : scores.get(0).keySet()){
			constraint = new Linear();
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			problem.add(constraint, ">=", 0);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		//System.out.println("Result : " + result);
		
		if(result == null){
//			System.out.println("Num of variables : " + problem.getVariablesCount());
//			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//			System.out.println("Objective Function : ");
//			System.out.println(problem.getObjective());
//			System.out.println("Constraints : ");
//			for(Constraint c : problem.getConstraints())
//				System.out.println(c);
			
			System.out.println("Result is NULL ... " + "...  Skipping this");
			
			return predictedVals;

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}
		
		//System.out.println(yPredicted);
		//System.exit(0);
		
		return predictedVals;
	}

	public YZPredicted generateYZPredictedILP_lagrangian(List<Counter<Integer>> scores,
			  int numOfMentions,
			  int nilIndex,
			  int [] yLabelsGold,
			  double Lambda_i[]){

		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		
		Linear objective = new Linear();
		
		////////////// Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);
				
				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		
		for(int label : scores.get(0).keySet()){
			if(label != nilIndex){
				String var = "y" + label;
				double coeff = Lambda_i[label];
				objective.add(coeff, var);
			}
		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		 
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			constraint = new Linear();
			for(int y : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int y : score.keySet()){
				constraint = new Linear();
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		//System.out.println("\n-----------------");
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		
		for(int y : scores.get(0).keySet()){
			constraint = new Linear();
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			problem.add(constraint, ">=", 0);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		//System.out.println("Result : " + result.getObjective());
		
		if(result == null){
//			System.out.println("Num of variables : " + problem.getVariablesCount());
//			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//			System.out.println("Objective Function : ");
//			System.out.println(problem.getObjective());
//			System.out.println("Constraints : ");
//			for(Constraint c : problem.getConstraints())
//				System.out.println(c);
			
			System.err.println("LossAugInf(hamming): Result is NULL ... " + "...  Skipping this");
			
			return predictedVals;

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}
		
		//System.out.println(yPredicted);
		//System.exit(0);
		
		return predictedVals;
	}
	
	public YZPredicted generateYZPredictedILP_hammingloss(List<Counter<Integer>> scores,
			  int numOfMentions,
			  int nilIndex,
			  int [] yLabelsGold){

		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		
		Linear objective = new Linear();
		
		////////////// Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);
				
				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		
		for(int label : scores.get(0).keySet()){
			// Implementation of hamming loss function
			String var = "y" + label;
			double coeff = 0.0;
			
			// (1 - 2 gold_ylabel) * ylabel
			if(label == nilIndex ){
				if(yLabelsGold.length == 0) 
					coeff = -1;
				else
					coeff = 1;
			}
			else {
				for(int ygold_label : yLabelsGold){
					if(ygold_label == label)
						coeff = -1;
				}
				if(coeff != -1)
					coeff = 1;
			}
			objective.add(coeff, var);	
		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		 
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			constraint = new Linear();
			for(int y : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int y : score.keySet()){
				constraint = new Linear();
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		//System.out.println("\n-----------------");
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		
		for(int y : scores.get(0).keySet()){
			constraint = new Linear();
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			problem.add(constraint, ">=", 0);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		//System.out.println("Result : " + result);
		
		if(result == null){
//			System.out.println("Num of variables : " + problem.getVariablesCount());
//			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//			System.out.println("Objective Function : ");
//			System.out.println(problem.getObjective());
//			System.out.println("Constraints : ");
//			for(Constraint c : problem.getConstraints())
//				System.out.println(c);
			
			System.err.println("LossAugInf(hamming): Result is NULL ... " + "...  Skipping this");
			
			return predictedVals;

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}
		
		//System.out.println(yPredicted);
		//System.exit(0);
		
		return predictedVals;
	}
	
	public YZPredicted generateYZPredictedILP(List<Counter<Integer>> scores,
												  int numOfMentions,
												  int nilIndex){
		
		YZPredicted predictedVals = new YZPredicted(numOfMentions);
		
		Counter<Integer> yPredicted = predictedVals.getYPredicted();
		int [] zPredicted = predictedVals.getZPredicted();
		
		//System.out.println("Calling ILP inference for Pr (Y,Z | X,T)");
		
		SolverFactory factory = new SolverFactoryLpSolve();
		factory.setParameter(Solver.VERBOSE, 0);
		factory.setParameter(Solver.TIMEOUT, 100); // set timeout to 100 seconds

		Problem problem = new Problem();
		
		Linear objective = new Linear();
		
		////////////// Objective --------------------------------------
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int label : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+label;
				double coeff = score.getCount(label);
				objective.add(coeff, var);
				
				//System.out.print(score.getCount(label) + "  " + "z"+mentionIdx+"_"+"y"+label + " + ");
			}
		}
		/**
		 * Commenting to simulate huffmann
		 */
		//System.out.println();
//		for(String yLabel : yLabelIndex){
//			int y = yLabelIndex.indexOf(yLabel);
//			String var = "y"+y;
//			double coeff = typeBiasScores.getCount(y);
//			objective.add(coeff, var);
//			
//			//System.out.print(typeBiasScores.getCount(y)+" y"+y+" + ");
//		}
		
		problem.setObjective(objective, OptType.MAX);
		/////////// -----------------------------------------------------
		 
		//System.out.println("\n-----------------");
		/////////// Constraints ------------------------------------------

		/// 1. equality constraints \Sum_i z_ji = 1 \forall j
		Linear constraint;
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			constraint = new Linear();
			for(int y : score.keySet()){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");				
			}
			
			problem.add(constraint, "=", 1); // NOTE : To simulate Hoffmann 
			//System.out.println(" 0 = "+ "1");
		}
		
		//System.out.println("\n-----------------");
		
		/// 2. inequality constraint -- 1 ... z_ji <= y_i \forall j,i
		for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
			Counter<Integer> score = scores.get(mentionIdx);
			for(int y : score.keySet()){
				constraint = new Linear();
				String var1 = "z"+mentionIdx+"_"+"y"+y;
				String var2 = "y"+y;
				constraint.add(1, var1);
				constraint.add(-1, var2);
				problem.add(constraint, "<=", 0);
				//System.out.println("z"+mentionIdx+"_"+"y"+y +" - " + "y"+y + " <= 0");
			}
		}
		
		//System.out.println("\n-----------------");
		/// 3. inequality constraint -- 2 ... y_i <= \Sum_j z_ji \forall i
		/////////// ------------------------------------------------------
		
		for(int y : scores.get(0).keySet()){
			constraint = new Linear();
			for(int mentionIdx = 0; mentionIdx < numOfMentions; mentionIdx ++){
				String var = "z"+mentionIdx+"_"+"y"+y;
				constraint.add(1, var);
				//System.out.print("z"+mentionIdx+"_"+"y"+y + " + ");
			}
			constraint.add(-1, "y"+y);
			problem.add(constraint, ">=", 0);
			//System.out.println(" 0 - " + "y"+y +" >= 0" );
		}
		
		// Set the types of all variables to Binary
		for(Object var : problem.getVariables())
			problem.setVarType(var, Boolean.class);
		
//		System.out.println("Num of variables : " + problem.getVariablesCount());
//		System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//		System.out.println("Objective Function : ");
//		System.out.println(problem.getObjective());
//		System.out.println("Constraints : ");
//		for(Constraint c : problem.getConstraints())
//			System.out.println(c);
		
		// Solve the ILP problem by calling the ILP solver
		Solver solver = factory.get();
		Result result = solver.solve(problem);
		
		//System.out.println("Result : " + result);
		
		if(result == null){
//			System.out.println("Num of variables : " + problem.getVariablesCount());
//			System.out.println("Num of Constraints : " + problem.getConstraintsCount());
//			System.out.println("Objective Function : ");
//			System.out.println(problem.getObjective());
//			System.out.println("Constraints : ");
//			for(Constraint c : problem.getConstraints())
//				System.out.println(c);
			
			System.out.println("Result is NULL ... " + "...  Skipping this");
			
			return predictedVals;

		}
		
		for(Object var : problem.getVariables()) {
			if(result.containsVar(var) && (result.get(var).intValue() == 1)){
				if(var.toString().startsWith("y")) {
					//System.out.println(var + " = " + result.get(var) + " : Y-vars");
					int y = Integer.parseInt(var.toString().substring(1));
					if(y != nilIndex)
						yPredicted.setCount(y, result.get(var).doubleValue());
				}
				else if(var.toString().startsWith("z")) { 
					String [] split = var.toString().split("_");
					//System.out.println(split[0]);
					int mentionIdx = Integer.parseInt(split[0].toString().substring(1));
					//System.out.println(split[1]);
					int ylabel = Integer.parseInt(split[1].toString().substring(1));
					zPredicted[mentionIdx] = ylabel;
					
					//System.out.println(var + " = " + result.get(var) + " : Z-vars");
					
				}
			}
		}
		
		//System.out.println(yPredicted);
		//System.exit(0);
		
		return predictedVals;
	}

}
