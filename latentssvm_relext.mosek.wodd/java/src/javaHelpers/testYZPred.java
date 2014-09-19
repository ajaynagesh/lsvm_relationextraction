package javaHelpers;

import ilpInference.YZPredicted;

public class testYZPred {

	static void call(YZPredicted yz){
		int z[] = yz.getZPredicted();
		z[0] = 3507;
		z[1] = 101;
		z[2] = 201;
		z[3] = 501;
	}
	
	public static void main(String args[]){
		YZPredicted yz = new YZPredicted(4);

		call(yz);
		
		for (int i = 0; i < yz.getZPredicted().length; i++){
			System.out.println("z["+i + "] = " + yz.getZPredicted()[i]);
		}
		
		
	}
}
