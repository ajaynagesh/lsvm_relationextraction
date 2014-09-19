import java.util.concurrent.Callable;


public class ThreadEg implements Callable<Double> {

    private String command;

    public ThreadEg(String s){
        this.command=s;
    }

    private void processCommand(double timeout) {
        try {
        	long t = (long)timeout;
            Thread.sleep(t);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String toString(){
        return this.command;
    }

	@Override
	public Double call() throws Exception {
		// TODO Auto-generated method stub
        System.out.println(Thread.currentThread().getName() +" Start. Command = "+command);
        processCommand(Double.parseDouble(command) * 1/10);
        System.out.println(Thread.currentThread().getName()+" End.");
		return Double.parseDouble(command) * 10;
	}
}