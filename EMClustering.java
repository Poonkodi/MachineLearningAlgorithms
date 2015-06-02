
import java.util.Random;

import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class EMClustering {

	// Loads the initial values used in the text-book.
	public static MultivariateNormalDistribution[] loadDefaultClusters() {
		MultivariateNormalDistribution[] theta = {
				new MultivariateNormalDistribution(new double[]{2.5,  65.0},
						new double[][]{{1.0, 5.0},  { 5.0, 100.0}}),				
				new MultivariateNormalDistribution(new double[]{3.5,  70.0}, 
						new double[][]{{2.0, 10.0}, {10.0, 200.0}})};
		return theta;
	}
	
	// Loads the initial values used in the text-book.
	public static MultivariateNormalDistribution[] loadDefaultClusters2() {
		MultivariateNormalDistribution[] theta = {
				new MultivariateNormalDistribution(new double[]{2.5,  15.0}, 
						new double[][]{{1.0, 5.0},  { 5.0, 100.0}}),				
				new MultivariateNormalDistribution(new double[]{13.5,  70.0}, 
						new double[][]{{2.0, 10.0}, {10.0, 200.0}})};
		return theta;
	}
	
	//
	public static MultivariateNormalDistribution[] loadDefaultClusters(int k) {
		
		if (k == 3) {
		MultivariateNormalDistribution[] theta = {
				new MultivariateNormalDistribution(new double[]{2.5,  15.0}, 
						new double[][]{{1.0, 5.0},  { 5.0, 100.0}}),
				new MultivariateNormalDistribution(new double[]{13.5,  70.0},
						new double[][]{{2.0, 10.0}, {10.0, 200.0}}),
				new MultivariateNormalDistribution(new double[]{2.5,  15.0}, 
						new double[][]{{1.0, 0.6}, {1.0, 1.0}})};
		return theta;
		}
		
			return loadDefaultClusters2();
		
	}
	// Loads random values.
	public static MultivariateNormalDistribution[] loadRandomClusters() {

		Random rnd = new Random();
		
		MultivariateNormalDistribution[] theta = {
				new MultivariateNormalDistribution(new double[]{Math.abs(rnd.nextFloat()),  
						Math.abs(rnd.nextFloat())}, 
						new double[][]{{rnd.nextFloat(), rnd.nextFloat()},  { rnd.nextFloat(),rnd.nextFloat()}}),				
				new MultivariateNormalDistribution(new double[]{rnd.nextFloat(), 
						rnd.nextFloat()}, 
						new double[][]{{rnd.nextFloat(), rnd.nextFloat()},  { rnd.nextFloat(),rnd.nextFloat()}})};
		
		return theta;
	}
	
	public static void printProbs(double[][] p) {
		for (int i = 0; i < p[0].length; i++) {
			for (int j = 0; j < p.length; j++) {
				System.out.format("p_%d_%d = %2.5f\n", j, i, p[j][i]);
			}
		}
	}

	public static double[][] clusterWithEM(Point[] points, int k) {

		Random rnd = new Random();

		int MAX_ITERS = 10;

		//Point[] points = Point.loadOldFaithfulData();
		int num_points = points.length;

		//MultivariateNormalDistribution[] theta = loadDefaultClusters();
		//MultivariateNormalDistribution[] theta = loadDefaultClusters2();
		MultivariateNormalDistribution[] theta = loadDefaultClusters();


		int num_clusters = theta.length;
		
		for(int j=0;j<num_clusters;j++) {
			System.out.format("Init Mean %d : { %1.4f, %1.4f } \n", j, theta[j].getMeans()[0], 
					theta[j].getMeans()[1]);
			System.out.format("Init Covar %d : %s\n\n", j, theta[j].getCovariances());
		}
		double p[][]=new double[num_clusters][num_points];		

		double[] tau = new double[k];
		if (k == 2) {
			tau = new double[]{0.6, 0.4};
		} else {
			for (int i = 0; i < k; i++) {
				tau[i] = rnd.nextFloat();
			}
		}
		
		// Iterations
		for (int iter = 0; iter < MAX_ITERS; iter++) {
			// E step
			for (int i = 0; i < num_points; i++){
				double f[]=new double[num_clusters];
				double denom=0;

				for (int j = 0; j < num_clusters; j++) {
					f[j] = theta[j].density(new double[]{points[i].getX(), points[i].getY()});
					denom+=tau[j]*f[j];
				}
				for (int j = 0; j < num_clusters; j++) {
					p[j][i]=tau[j]*f[j]/denom;
				}
			}

			if (iter <= 2) {
				System.out.println("\nIter " + iter);
				printProbs(p);
			}

			// M step
			for(int j=0;j<num_clusters;j++)
			{
				double sum_p=0,sum_p_x=0,sum_p_y=0;

				double[] cur_mean = theta[j].getMeans();
				RealMatrix mean_mat = MatrixUtils.createColumnRealMatrix(cur_mean);
				RealMatrix covar_sum = MatrixUtils.createRealMatrix(2,2);

				for(int i=0;i<num_points;i++)
				{
					sum_p += p[j][i];
					sum_p_x += p[j][i]*points[i].getX();
					sum_p_y += p[j][i]*points[i].getY();

					// Compute covar;
					double[] cur_point = {points[i].getX(), points[i].getY()};

					//points[i].Print();
					RealMatrix pt = MatrixUtils.createColumnRealMatrix(cur_point);					
					RealMatrix diff = pt.subtract(mean_mat);

					covar_sum = covar_sum.add(diff.multiply(diff.transpose()).scalarMultiply(p[j][i]));
				}
				tau[j]=(1.0/num_clusters)*sum_p;

				//compute mean , covariance
				double mu_x=sum_p_x/sum_p;
				double mu_y=sum_p_y/sum_p;
				double [] new_mean = {mu_x,mu_y};

				RealMatrix new_cov = covar_sum.scalarMultiply(1.0/sum_p);
				
				System.out.format("Tau %d : { %1.4f } \n", j, tau[j]);
				System.out.format("Mean %d : { %1.4f, %1.4f } \n", j, mu_x, mu_y);
				System.out.format("Covar %d : %s\n\n", j, new_cov);

				theta[j] = new MultivariateNormalDistribution(new_mean, new_cov.getData());

				
			}
		}

		System.out.println("After iteration " + MAX_ITERS + ":");
		for (int j = 0; j < num_clusters; j++) {
			System.out.format("Mean %d : { %1.4f, %1.4f } \n", j, theta[j].getMeans()[0], 
					theta[j].getMeans()[1]);
			System.out.format("Covar %d : %s\n\n", j, theta[j].getCovariances());
		}
		return p;
	}
	
	public static void main(String[] args) {
		Point[] points = Point.loadPointsInBook();
		//Point[] points = Point.loadOldFaithfulData();

		double[][] memberships = clusterWithEM(points, 2);


	}
	
}









