
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;

public class SimpleThreadPool {
	
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        //ExecutorService executor = Executors.newFixedThreadPool(numthreads);
    	List<Future<Double>> futures = new ArrayList<Future<Double>>(10);
    	ExecutorService executor = Executors.newCachedThreadPool();
    	
    	ListeningExecutorService l_executor = MoreExecutors.listeningDecorator(executor);
    	
        for (int i = 0; i < 3; i++) {
            Callable<Double> worker = new ThreadEg( " " + i);
//            Future<Double> res = executor.submit(worker);
            ListenableFuture<Double> res = l_executor.submit(worker);
            Futures.transform(res, new AsyncFunction<Double, Double>() {

				@Override
				public ListenableFuture<Double> apply(Double arg0)
						throws Exception {
					// TODO Auto-generated method stub
					return Futures.immediateFuture(arg0);
				}
            	
			}, l_executor);
            futures.add(res);
          }
        
        
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");
        System.out.println("Printing return vals");
        for(Future<Double> future : futures){
        	System.out.println(future.get());
        }
    }

}