package com.github.chen0040.lof;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;


/**
 * Created by memeanalytics on 17/8/15.
 */
public class DistanceMeasureService {
    public static double getDistance(DataFrame mgr, DataRow t1, DataRow t2, BiFunction<DataRow, DataRow, Double> distance){
        if(distance== null){
            double[] x1 = t1.toArray();
            double[] x2 = t2.toArray();
            return euclideanDistance(x1, x2);
        }else{
            return distance.apply(t1, t2);
        }
    }

    public static double euclideanDistance(double[] x1, double[] x2){
        int dimension = Math.min(x1.length, x2.length);
        double cross_prod = 0;
        for(int i=0; i < dimension; ++i){
            cross_prod += (x1[i]-x2[i]) * (x1[i]-x2[i]);
        }
        return Math.sqrt(cross_prod);
    }

    public static Map<DataRow, Double> getKNearestNeighbors(DataFrame batch, DataRow t, int k, BiFunction<DataRow, DataRow, Double> distanceMeasure){
        Map<DataRow, Double> neighbors = new HashMap<DataRow, Double>();
        for(int i = 0; i < batch.rowCount(); ++i){

            DataRow ti = batch.row(i);
            if(ti == t) continue;
            double distance = getDistance(batch, ti, t, distanceMeasure);
            if(neighbors.size() < k){
                neighbors.put(ti, distance);
            }else{
                double largest_distance = Double.MIN_VALUE;
                DataRow neighbor_with_largest_distance = null;
                for(DataRow tj : neighbors.keySet()){
                    double tj_distance = neighbors.get(tj);
                    if(tj_distance > largest_distance){
                        largest_distance =tj_distance;
                        neighbor_with_largest_distance = tj;
                    }
                }

                if(largest_distance > distance){
                    neighbors.remove(neighbor_with_largest_distance);
                    neighbors.put(ti, distance);
                }
            }
        }

        return neighbors;
    }

    public static Object[] getKthNearestNeighbor(DataFrame batch, DataRow tuple, int k, BiFunction<DataRow, DataRow, Double> distanceMeasure) {
        Map<DataRow,Double> neighbors = getKNearestNeighbors(batch, tuple, k, distanceMeasure);

        double largest_distance = Double.MIN_VALUE;
        DataRow neighbor_with_largest_distance = null;
        for(DataRow tj : neighbors.keySet()){
            double tj_distance = neighbors.get(tj);
            if(tj_distance > largest_distance){
                largest_distance =tj_distance;
                neighbor_with_largest_distance = tj;
            }
        }

        return new Object[] {neighbor_with_largest_distance, largest_distance};
    }
}
