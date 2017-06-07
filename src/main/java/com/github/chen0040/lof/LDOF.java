package com.github.chen0040.lof;

import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.TupleTwo;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.*;
import java.util.function.BiFunction;
import java.util.stream.Collectors;


/**
 * Created by memeanalytics on 19/8/15.
 */
@Setter
@Getter
public class LDOF {

    private BiFunction<DataRow, DataRow, Double> distanceMeasure;
    private int minPts = 5; // k, namely the number of points in k-nearest neighborhood
    private int anomalyCount = 10; // number of outliers to row

    @Setter(AccessLevel.NONE)
    private int ldofLB; // lowerbounds for lodf, for pruning purpose

    @Setter(AccessLevel.NONE)
    private DataFrame model;

    public LDOF(){
        super();
        minPts = 5;
        anomalyCount = 10;
    }

    public int getLdofLB() {
        return ldofLB;
    }

    public void setLdofLB(int ldofLB) {
        this.ldofLB = ldofLB;
    }

    public BiFunction<DataRow, DataRow, Double> getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(BiFunction<DataRow, DataRow, Double> distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public int getMinPts() {
        return minPts;
    }

    public void setMinPts(int minPts) {
        this.minPts = minPts;
    }

    public double knn_distance(DataRow o, List<TupleTwo<DataRow, Double>> result){
        double avg_distance = 0;
        for(TupleTwo<DataRow, Double> d : result) {
            avg_distance += d._2();
        }
        avg_distance /= result.size();

        return avg_distance;
    }



    public double knn_inner_distance(DataFrame context, DataRow o, List<TupleTwo<DataRow, Double>> result){
        List<DataRow> nn = result.stream().map(TupleTwo::_1).collect(Collectors.toList());

        double distance_sum = 0;
        for(int i=0; i < nn.size(); ++i){
            for(int j=i+1; j < nn.size(); ++j){
                DataRow ti = nn.get(i);
                DataRow tj = nn.get(j);
                distance_sum += DistanceMeasureService.getDistance(context, ti, tj, distanceMeasure);
            }
        }
        distance_sum *= 2; //because of symmetry

        double avg_distance = distance_sum / ((nn.size()-1) * nn.size());

        return avg_distance;
    }

    public double local_distance_outlier_factor(DataFrame batch, DataRow p, int k){
        List<TupleTwo<DataRow, Double>> result = DistanceMeasureService.getKNearestNeighbors(batch, p, k, distanceMeasure);
        double knn_distance = knn_distance(p, result);
        double knn_inner_distance = knn_inner_distance(batch, p, result);

        return knn_distance / knn_inner_distance;
    }



    public DataFrame getModel(){
        return model;
    }

    public DataFrame fitAndTransform(DataFrame batch) {
        this.model = batch.makeCopy();

        List<DataRow> maybeoutliers = new ArrayList<DataRow>();
        final HashMap<DataRow, Double> ldof_scores = new HashMap<DataRow, Double>();

        int m = model.rowCount();
        for(int i=0; i < m; ++i){
            DataRow tuple = model.row(i);
            tuple.setCategoricalTargetCell("anomaly", "0");

            double ldof = local_distance_outlier_factor(model, tuple, minPts);

            if(ldof >= ldofLB){
                maybeoutliers.add(tuple);
                ldof_scores.put(tuple, ldof);
            }
        }

        // sort descendingly based on the ldof score
        Collections.sort(maybeoutliers, (o1, o2) -> {
            double ldof1 = ldof_scores.get(o1);
            double ldof2 = ldof_scores.get(o2);
            return Double.compare(ldof2, ldof1);
        });

        for(int i=0; i < anomalyCount && i < maybeoutliers.size(); ++i){
            maybeoutliers.get(i).setCategoricalTargetCell("anomaly", "1");
        }

        return this.getModel();
    }
}
