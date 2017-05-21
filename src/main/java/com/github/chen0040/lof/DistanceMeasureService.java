package com.github.chen0040.lof;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.TupleTwo;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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

    public static List<TupleTwo<DataRow, Double>> getKNearestNeighbors(DataFrame batch, DataRow t, int k, BiFunction<DataRow, DataRow, Double> distanceMeasure){

        MinPQ<DataRow> minPQ = new MinPQ<>();

        int N = batch.rowCount();
        for(int i = 0; i < N; ++i){

            DataRow ti = batch.row(i);
            if(ti == t) continue;
            double distance = getDistance(batch, ti, t, distanceMeasure);
            minPQ.enqueue(ti, distance);
        }

        List<TupleTwo<DataRow, Double>> neighbors = new ArrayList<>();
        int m = Math.min(k, N);
        for(int i=0; i < m; ++i){
            TupleTwo<DataRow, Double> tuple = minPQ.delMin();
            neighbors.add(tuple);
        }

        return neighbors;
    }

    public static TupleTwo<DataRow, Double> getKthNearestNeighbor(DataFrame batch, DataRow tuple, int k, BiFunction<DataRow, DataRow, Double> distanceMeasure) {
        List<TupleTwo<DataRow,Double>> neighbors = getKNearestNeighbors(batch, tuple, k, distanceMeasure);

        return neighbors.get(neighbors.size()-1);
    }
}
