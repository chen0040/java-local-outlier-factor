package com.github.chen0040.lof;


import com.github.chen0040.data.utils.TupleTwo;
import org.testng.annotations.Test;

import java.util.Random;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.testng.Assert.*;


/**
 * Created by xschen on 21/5/2017.
 */
public class MinPQUnitTest {

   @Test
   public void test_min_pq(){
      MinPQ<Integer> minPQ = new MinPQ<>();

      Random random = new Random();
      for(int i=0; i < 100; ++i) {
         minPQ.enqueue(i, random.nextDouble());
      }

      double prevCost = Double.NEGATIVE_INFINITY;
      for(int i=0; i < 100; ++i){
         TupleTwo<Integer, Double> item = minPQ.delMin();
         double cost = item._2();
         assertThat(cost).isGreaterThanOrEqualTo(prevCost);
         prevCost = cost;
      }
   }

}
