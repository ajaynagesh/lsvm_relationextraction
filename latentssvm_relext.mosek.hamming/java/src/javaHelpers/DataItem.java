package javaHelpers;

import java.util.ArrayList;
import edu.stanford.nlp.stats.Counter;

public class DataItem {
	public DataItem(int numYlabels) {
		super();
		this.pattern = new ArrayList<Counter<Integer>>();
		this.ylabel = new int[numYlabels];
	}

	public ArrayList<Counter<Integer>> pattern; // mention
	public int [] ylabel; // relation
}

