import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


public class LossSurfacePoints {

	public static void main(String args[]) throws IOException{
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("loss_points_all.out")));
		int Np = 4700, Nn = 65000;
		for(int FP = 0; FP <= Nn; FP ++){
			for (int FN = 0; FN <= Np; FN ++){
				double loss = ((double)( FP + FN)) / (double)(2*Np + FP - FN);
				bw.write(FP + "\t" + FN + "\t" + loss + "\n");
			}
		}
		bw.close();
	}
	
}
