import java.util.Random;

public class Kmeans 
{
	public static double distance(Point p1, Point p2) {
		return Math.sqrt(
				Math.pow(p1.getX()-p2.getX(),2)+
				Math.pow(p1.getY()-p2.getY(),2));				
	}

	public static int[] computeKMeans(Point[] points, int k) {
		
		Random rnd = new Random();
		
		int MAX_ITERS = 100;
		Point[] centroids = new Point[k];
		for (int j = 0; j < k; j++) {
			centroids[j] = new Point();
		}

		int num_points = points.length;

		int[] memberships = new int[num_points];

		// Assign alternating points to clusters.
		for (int i = 0; i < num_points; i++) {
				memberships[i] = rnd.nextInt(k);
		}

		for (int iter = 0; iter < MAX_ITERS; iter++) {

			// Compute centroids
			for (int j = 0; j < k; j++) {
				double sum_x = 0;
				double sum_y = 0;
				int count = 0;
				for (int i = 0; i < num_points; i++) {
					if (memberships[i] == j) {
						sum_x += points[i].getX();
						sum_y += points[i].getY();
						count++;							
					}
				}
				centroids[j].setX(sum_x / count);
				centroids[j].setY(sum_y / count);
				
				//System.out.println("Centroid " + j + " : " + centroids[j]);
			}

			// Re-assign points
			for (int i = 0; i < num_points; i++) {
				double min_dist = Double.MAX_VALUE;
				int new_cluster = -1;
				for (int j = 0; j < k; j++) {
					double dist = distance(points[i], centroids[j]);
					if (dist < min_dist) {
						min_dist = dist;
						new_cluster = j;
					}
				}
				if (new_cluster == -1 ) {
					throw new IllegalStateException("Illegal cluster id ");
				}
				memberships[i] = new_cluster;
			}
		}
		for (int j = 0; j < k; j++) {
			System.out.println("Centroid " + j + " : " + centroids[j]);
		}
		System.out.println();
		
		return memberships;
	}

	public static void main(String args[]) {

		Point[] points = Point.loadOldFaithfulData();
		
		int[] kmeans_mem = computeKMeans(points, 2);
		int[] kmeans_mem2 = computeKMeans(points, 3);
		int[] kmeans_mem3 = computeKMeans(points, 4);
//		double[][] em_mem = EMClustering.clusterWithEM(points, 2);
//
//		System.out.println("id, x, y, kmeans, em_1, em2");
//		for (int i = 0; i < points.length; i++) {
//			System.out.format("%d, %3.4f, %3.4f, %d, %3.4f, %3.4f\n", i, points[i].getX(), points[i].getY(), 
//					kmeans_mem[i], em_mem[0][i], em_mem[1][i]);
//		}
//		computeKMeans(points, 2);
//		computeKMeans(points, 2);
//		computeKMeans(points, 10);
		
	}
}

