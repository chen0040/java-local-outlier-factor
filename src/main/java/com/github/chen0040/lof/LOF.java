package com.github.chen0040.lof;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import lombok.Getter;
import lombok.Setter;

import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 * Created by memeanalytics on 17/8/15.
 * Link:
 * http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf
 */
@Getter
@Setter
public class LOF {

    public double threshold = 0.5;

    // min number for minPts;
    public int minPtsLB = 3;

    // max number for minPts;
    public int minPtsUB = 10;
    public boolean parallel = true;
    public boolean automaticThresholding = true;
    public double automaticThresholdingRatio = 0.05;


    private static final Logger logger = Logger.getLogger(String.valueOf(LOF.class));

    public BiFunction<DataRow, DataRow, Double> distanceMeasure;
    private double minScore;
    private double maxScore;
    private boolean addPredictedLabelAfterBatchUpdate = false;
    private DataFrame model;


    protected void adjustThreshold(DataFrame batch){
        int m = batch.rowCount();

        List<Integer> orders = new ArrayList<Integer>();
        List<Double> probs = new ArrayList<Double>();

        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            double prob = evaluate(tuple, model);
            probs.add(prob);
            orders.add(i);
        }

        final List<Double> probs2 = probs;
        // sort descendingly by probability values
        Collections.sort(orders, new Comparator<Integer>() {
            public int compare(Integer h1, Integer h2) {
                double prob1 = probs2.get(h1);
                double prob2 = probs2.get(h2);
                return Double.compare(prob2, prob1);
            }
        });

        int selected_index = autoThresholdingCaps(orders.size());
        if(selected_index >= orders.size()){
            threshold = probs.get(orders.get(orders.size() - 1));
        }
        else{
            threshold = probs.get(orders.get(selected_index));
        }

    }

    public LOF(){
        super();
        threshold = 0.5;
        setSearchRange(3, 10);
        parallel = true;
        automaticThresholding = true;
        automaticThresholdingRatio = 0.05;
    }

    protected int autoThresholdingCaps(int m){
        return Math.max(1, (int) (automaticThresholdingRatio * m));
    }

    public double getMinScore() {
        return minScore;
    }

    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }

    public double getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(double maxScore) {
        this.maxScore = maxScore;
    }

    public DataFrame getModel(){
        return model;
    }

    public void copy(LOF that){
        minScore = that.minScore;
        maxScore = that.maxScore;
        distanceMeasure = that.distanceMeasure;
        addPredictedLabelAfterBatchUpdate = that.addPredictedLabelAfterBatchUpdate;
        model = that.model == null ? null : that.model.makeCopy();
    }

    public LOF makeCopy(){
        LOF clone = new LOF();
        clone.copy(this);

        return clone;
    }

    public MinPtsBounds searchRange() {
        return new MinPtsBounds(minPtsLB, minPtsUB);
    }

    public void setSearchRange(int minPtsLB, int minPtsUB) {
        this.minPtsLB = minPtsLB;
        this.minPtsUB = minPtsUB;
    }

    public BiFunction<DataRow, DataRow, Double> getDistanceMeasure() {
        return distanceMeasure;
    }

    public void setDistanceMeasure(BiFunction<DataRow, DataRow, Double> distanceMeasure) {
        this.distanceMeasure = distanceMeasure;
    }

    public boolean isAnomaly(DataRow tuple) {
        double score_lof = evaluate(tuple, model);
        return score_lof > threshold;
    }

    private class ScoreTask implements Callable<Double>{
        private DataFrame batch;
        private DataRow tuple;
        public ScoreTask(DataFrame batch, DataRow tuple){
            this.batch = batch;
            this.tuple = tuple;
        }

        public Double call() throws Exception {
            double score = score_lof_sync(batch, tuple);
            return score;
        }
    }



    public void fit(DataFrame batch) {
        this.model = batch.makeCopy();

        int m = model.rowCount();

        minScore = Double.MAX_VALUE;
        maxScore = Double.NEGATIVE_INFINITY;



        if(parallel) {
            ExecutorService executor = Executors.newFixedThreadPool(10);
            List<ScoreTask> tasks = new ArrayList<>();
            for (int i = 0; i < m; ++i) {
                tasks.add(new ScoreTask(model, model.row(i)));
            }

            try {
                List<Future<Double>> results = executor.invokeAll(tasks);
                executor.shutdown();
                for (int i = 0; i < m; ++i) {
                    double score = results.get(i).get();
                    if(Double.isNaN(score)) continue;
                    if(Double.isInfinite(score)) continue;
                    minScore = Math.min(score, minScore);
                    maxScore = Math.max(score, maxScore);
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }else{
            for(int i=0; i < m; ++i){
                double score = score_lof_sync(model, model.row(i));
                if(Double.isNaN(score)) continue;
                if(Double.isInfinite(score)) continue;
                minScore = Math.min(score, minScore);
                maxScore = Math.max(score, maxScore);
            }
        }

        if(automaticThresholding){
            adjustThreshold(model);
        }

        if(addPredictedLabelAfterBatchUpdate){
            for(int i=0; i < m; ++i){
                DataRow tuple = model.row(i);
                double score_lof = evaluate(tuple, batch);
                tuple.setCategoricalTargetCell("label", score_lof > threshold ? "OUTLIER" : "NORMAL");
            }
        }
    }

    private class LOFTask implements Callable<Double>{
        private DataFrame batch;
        private DataRow tuple;
        private int minPts;

        public LOFTask(DataFrame batch, DataRow tuple, int minPts){
            this.batch = batch;
            this.tuple = tuple;
            this.minPts = minPts;
        }

        public Double call() throws Exception {
            double lof = local_outlier_factor(batch, tuple, minPts);
            return lof;
        }
    }

    private double score_lof_sync(DataFrame batch, DataRow tuple){
        double maxLOF = Double.NEGATIVE_INFINITY;

        for(int minPts = minPtsLB; minPts <= minPtsUB; ++minPts) { // the number of nearest neighbors used in defining the local neighborhood of the object.
            double lof = local_outlier_factor(batch, tuple, minPts);
            if(Double.isNaN(lof)) continue;
            maxLOF = Math.max(maxLOF, lof);
        }


        return maxLOF;
    }

    private double score_lof_async(DataFrame batch, DataRow tuple){
        if(!parallel){
            return score_lof_sync(batch, tuple);
        }

        double maxLOF = 0;

        ExecutorService executor = Executors.newFixedThreadPool(Math.min(8, minPtsUB - minPtsLB + 1));

        List<LOFTask> tasks = new ArrayList<LOFTask>();
        for(int minPts = minPtsLB; minPts <= minPtsUB; ++minPts) { // the number of nearest neighbors used in defining the local neighborhood of the object.
            tasks.add(new LOFTask(batch, tuple, minPts));
        }

        try {
            List <Future<Double>> results = executor.invokeAll(tasks);
            executor.shutdown();
            for(int i=0; i < results.size(); ++i){
                double lof = results.get(i).get();
                if(Double.isNaN(lof)) continue;
                if(Double.isInfinite(lof)) continue;
                maxLOF = Math.max(maxLOF, lof);
            }
        } catch (InterruptedException | ExecutionException e) {
            logger.log(Level.SEVERE, "score_lof_async failed", e);
        }

        return maxLOF;
    }

    public double evaluate(DataRow tuple, DataFrame context){
        double score = score_lof_async(model, tuple);

        //logger.info(String.format("score: %f minScore: %f, maxScore: %f", score, minScore, maxScore));

        score -= minScore;
        if(score < 0) score = 0;

        score /= (maxScore - minScore);

        if(score > 1) score = 1;

        return score;
    }

    private double evaluate_sync(DataRow tuple, DataFrame batch){
        double score = score_lof_sync(batch, tuple);

        score -= minScore;
        if(score < 0) score = 0;

        score /= (maxScore - minScore);

        if(score > 1) score = 1;

        return score;
    }

    public double k_distance(DataFrame batch, DataRow o, int k){
        Object[] result = DistanceMeasureService.getKthNearestNeighbor(batch, o, k, distanceMeasure);
        //DataRow o_k = (DataRow)result[0];
        double k_distance = (Double)result[1];
        return k_distance;
    }

    private double reach_dist(DataFrame batch, DataRow p, DataRow o, int k){
        double distance_p_o = DistanceMeasureService.getDistance(batch, p, o, distanceMeasure);
        double distance_k_o = k_distance(batch, o, k);
        return Math.max(distance_k_o, distance_p_o);
    }

    private double local_reachability_density(DataFrame batch, DataRow p, int k){
        Map<DataRow, Double> knn_p = DistanceMeasureService.getKNearestNeighbors(batch, p, k, distanceMeasure);
        double density = local_reachability_density(batch, p, k, knn_p);
        return density;
    }

    private double local_reachability_density(DataFrame batch, DataRow p, int k, Map<DataRow, Double> knn_p){
        double sum_reach_dist = 0;
        for(DataRow o : knn_p.keySet()){
            sum_reach_dist += reach_dist(batch, p, o, k);
        }
        double density = 1 / (sum_reach_dist / knn_p.size());
        return density;
    }

    // the higher this value, the more likely the point is an outlier
    public double local_outlier_factor(DataFrame batch, DataRow p, int k){

        Map<DataRow, Double> knn_p = DistanceMeasureService.getKNearestNeighbors(batch, p, k, distanceMeasure);
        double lrd_p = local_reachability_density(batch, p, k, knn_p);
        double sum_lrd = 0;
        for(DataRow o : knn_p.keySet()){
            sum_lrd += local_reachability_density(batch, o, k);
        }

        if(Double.isInfinite(sum_lrd) && Double.isInfinite(lrd_p)){
            return 1 / knn_p.size();
        }

        double lof = (sum_lrd / lrd_p) / knn_p.size();

        return lof;
    }

    public class MinPtsBounds{
        private int lowerBound;
        private int upperBound;

        public void setLowerBound(int lowerBound) {
            this.lowerBound = lowerBound;
        }

        public void setUpperBound(int upperBound) {
            this.upperBound = upperBound;
        }

        public MinPtsBounds(int lowerBounds, int upperBounds){
            lowerBound = lowerBounds;
            upperBound = upperBounds;
        }

        public int getLowerBound(){
            return lowerBound;
        }

        public int getUpperBound(){
            return upperBound;
        }
    }

}
