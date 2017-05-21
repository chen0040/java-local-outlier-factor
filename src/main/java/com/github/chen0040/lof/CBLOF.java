package com.github.chen0040.lof;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataRow;
import com.github.chen0040.data.utils.CountRepository;
import com.github.chen0040.data.utils.discretizers.KMeansDiscretizer;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


/**
 * Created by memeanalytics on 18/8/15.
 * Link:
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.20.4242&rep=rep1&type=pdf
 */
@Getter
@Setter
public class CBLOF {

    private ArrayList<Cluster> clusters;

    @Setter(AccessLevel.NONE)
    private KMeansDiscretizer discretizer;

    private int split;

    public double threshold;
    public boolean parallel;
    public boolean automaticThresholding;
    public double anomalyRatioInAutomaticThresholding;
    public double similarityThreshold;
    public double alpha;
    public double beta;

    public void copy(CBLOF that){
        
        split = that.split;
        discretizer = that.discretizer == null ? null : that.discretizer.makeCopy();

        clusters = null;
        if(that.clusters != null){
            clusters = new ArrayList<>();
            for(int i=0; i < that.clusters.size(); ++i){
                clusters.add((Cluster)that.clusters.get(i).clone());
            }
        }
    }

    public CBLOF makeCopy(){
        CBLOF clone = new CBLOF();
        clone.copy(this);

        return clone;
    }

    public CBLOF(){
        super();
        KMeansDiscretizer d = new KMeansDiscretizer();
        d.setMaxLevelCount(10);
        discretizer = d;
        threshold = 0.5;
        alpha = 0.8;
        beta = 0.1;
        similarityThreshold =  0.8;
        parallel =true;
        automaticThresholding = false;
        anomalyRatioInAutomaticThresholding = 0.05;
    }

    public boolean isAnomaly(DataRow tuple) {
        double CBLOF = transform(tuple);
        return CBLOF > threshold;
    }

    protected void adjustThreshold(DataFrame batch){
        int m = batch.rowCount();

        List<Integer> orders = new ArrayList<Integer>();
        List<Double> probs = new ArrayList<Double>();

        for(int i=0; i < m; ++i){
            DataRow tuple = batch.row(i);
            double prob = transform(tuple);
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

        int selected_position = (int)(anomalyRatioInAutomaticThresholding * orders.size());


        int last_index = orders.get(orders.size() - 1);
        int selected_index = -1;
        if(selected_position >= orders.size()){
            selected_position = orders.size() - 1;
            selected_index = last_index; //setAttribute(THRESHOLD, probs.get(orders.get(orders.size() - 1)));
        }
        else {
            selected_index = orders.get(selected_position);
        }

        while(probs.get(selected_index) == probs.get(last_index) && selected_position > 0){
            selected_position--;
            selected_index = orders.get(selected_position);
        }

        threshold = probs.get(selected_index);
    }

    public DataFrame fitAndTransform(DataFrame frame) {
        DataFrame dataFrame = discretizer.fitAndTransform(frame);

        runSqueezer(dataFrame, similarityThreshold);

        // sort descendingly based on cluster size
        Collections.sort(clusters, (o1, o2) -> Integer.compare(o2.size(), o1.size()));

        for(int i=0; i < clusters.size(); ++i){
            Cluster cluster = clusters.get(i);
            cluster.setIndex(i);
            //System.out.println("cluster[" +i +"].size: " + cluster.size());
        }

        int m = dataFrame.rowCount();

        split = 0; // clusters with index < split will be the large clusters; otherwise small clusters
        int accumulated_count = 0;

        for(split = 0; split < clusters.size() - 1; ++split){
            int current_cluster_size = clusters.get(split).size();
            accumulated_count += current_cluster_size;
            if(accumulated_count >= m * alpha && split != 0) break;
            int next_cluster_size = clusters.get(split+1).size();
            double ratio = (double)current_cluster_size / next_cluster_size;
            if(ratio  < beta && split != 0) break;
        }

        //System.out.println("split: "+split);

        for(int i=0; i < m; ++i){
            DataRow row = dataFrame.row(i);
            Cluster c = clusters.get(Integer.parseInt(row.getCategoricalTargetCell("cluster")));

            if(c.getIndex() > split){ // c belongs to small clusters
                double minDistance = Double.MAX_VALUE;
                for(int j=0; j <= split; ++j){
                    double distance = clusters.get(j).distance(row);
                    if(minDistance > distance){
                        minDistance = distance;
                    }
                }

                row.setTargetCell("CBLOF", c.size() * minDistance);
            }else{
                row.setTargetCell("CBLOF", c.size() * c.distance(row));
            }

        }

        if(automaticThresholding){
            adjustThreshold(dataFrame);
        }

        return dataFrame;
    }

    private void runSqueezer(DataFrame batch, double s){
        clusters = new ArrayList<>();

        int m = batch.rowCount();
        for(int i=0; i < m; ++i){
            DataRow row = batch.row(i);

            if(i==0){
                clusters.add(new Cluster(row));
            }
            else{
                double maxSim = Double.MIN_VALUE;
                Cluster closestCluster = null;
                for(Cluster c : clusters){
                    double sim = c.similarity(row);
                    if(sim > maxSim){
                        maxSim = sim;
                        closestCluster = c;
                    }
                }

                if(maxSim < s){
                    clusters.add(new Cluster(row));
                }else{
                    closestCluster.add(row);
                }
            }


        }
    }



    // the higher the CBLOF, the more likely the tuple is an outlier

    public double transform(DataRow row) {
        row = discretizer.transform(row);
        double CBLOF;

        double maxSim = Double.MIN_VALUE;
        Cluster closestCluster = null;
        for(Cluster c : clusters){
            double sim = c.similarity(row);
            if(sim > maxSim){
                maxSim = sim;
                closestCluster = c;
            }
        }

        assert closestCluster != null;

        if(closestCluster.getIndex() > split){ // c belongs to small clusters
            double minDistance = Double.MAX_VALUE;
            for(int j=0; j <= split; ++j){
                double distance = clusters.get(j).distance(row);
                if(minDistance > distance){
                    minDistance = distance;
                }
            }

            CBLOF = closestCluster.size() * minDistance;
        }else{
            CBLOF = closestCluster.size() * closestCluster.distance(row);
        }

        return CBLOF;
    }

    private class Cluster implements Cloneable {
        private CountRepository counts;
        private int totalCount;
        private int index;

        public void copy(Cluster rhs){
            counts = rhs.counts.makeCopy();
            totalCount = rhs.totalCount;
            index = rhs.index;
        }

        @Override
        public Object clone(){
            Cluster clone = new Cluster();
            clone.copy(this);

            return clone;
        }

        public Cluster(){
            counts = new CountRepository();
            totalCount = 0;
        }

        public Cluster(DataRow tuple){
            counts = new CountRepository();
            totalCount = 0;
            add(tuple);
        }

        public int getIndex(){
            return this.index;
        }

        public void setIndex(int index){
            this.index = index;
        }

        public void add(DataRow row){

            List<String> columnNames = row.getCategoricalColumnNames();
            for(String columnName : columnNames){
                String value = row.getCategoricalCell(columnName);
                String eventName = columnName+"="+value;

                counts.addSupportCount(eventName);
                counts.addSupportCount(columnName);
            }
            row.setCategoricalTargetCell("cluster", "" + index);
            totalCount++;
        }

        public int size(){
            return totalCount;
        }

        public double similarity(DataRow row){

            double similarity = 0;
            List<String> columnNames = row.getCategoricalColumnNames();
            for(String columnName : columnNames){
                String value = row.getCategoricalCell(columnName);
                String eventName = columnName+"="+value;

                double count_Ai = counts.getSupportCount(eventName);
                double count_Total = counts.getSupportCount(columnName);


                if(count_Total > 0) {
                    similarity += count_Ai / count_Total;
                }
            }

            return similarity / columnNames.size();
        }

        public double distance(DataRow tuple){
            double sim = similarity(tuple);
            return 1 - sim;
        }
    }

}
