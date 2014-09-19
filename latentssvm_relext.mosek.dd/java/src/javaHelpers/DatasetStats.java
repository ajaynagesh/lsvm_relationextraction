package javaHelpers;

public class DatasetStats {
	
	/// Deafult: These are stats for Riedel dataset where only positive (non-nil) datapoints are considered
	
	int N;
	int Np;
	int L; 

	int maxFP; 
	int maxFN;
	
	public DatasetStats(int N, int Np, int L){
		this.N = N;
		this.Np = Np;
		this.L = L;
		this.maxFP = (N * L) - Np;
		this.maxFN = Np;
	}
	
	@Override
	public String toString() {
		return "Dataset Stats: N : " + N + "\tNp : " + Np + "\tL : " + L;
	}
	
}
