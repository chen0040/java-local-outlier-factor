package com.github.chen0040.lof;


import com.github.chen0040.data.utils.TupleTwo;


/**
 * Created by xschen on 21/5/2017.
 */
public class MinPQ<T> {
   public TupleTwo<T, Double>[] s;
   private int N = 0;

   public MinPQ(){
      s = (TupleTwo<T, Double>[])new TupleTwo[20];
   }

   public void enqueue(T item, double cost) {
      if(N+1 == s.length) resize(s.length * 2);
      s[++N] = new TupleTwo<>(item, cost);

      swim(N);
   }

   public TupleTwo<T, Double> delMin(){
      TupleTwo<T, Double> item = s[1];
      exchange(s, 1, N--);
      sink(1);
      if(N == s.length / 4) resize(s.length / 2);

      return item;
   }

   public int size(){
      return N;
   }

   private void swim(int k){
      while(k > 1){
         int parent = k / 2;
         if(less(s, k, parent)) {
            exchange(s, k, parent);
            k = parent;
         } else {
            break;
         }
      }
   }

   private void sink(int k) {
      while(k * 2 <= N){
         int child = k * 2;
         if(child < N && less(s, child+1, child)){
            child++;
         }
         if(less(s, child, k)){
            exchange(s, k, child);
            k = child;
         } else {
            break;
         }
      }
   }

   private static <T> void exchange(TupleTwo<T, Double>[] s, int i, int j){
      TupleTwo<T, Double> temp = s[i];
      s[i] = s[j];
      s[j] = temp;
   }

   private static <T> boolean less(TupleTwo<T, Double>[] s, int i, int j){
      return s[i]._2() < s[j]._2();
   }

   private void resize(int len){

      TupleTwo<T, Double>[] temp = (TupleTwo<T, Double>[])new TupleTwo[len];

      len = Math.min(len, s.length);
      for(int i=0; i < len; ++i){
         temp[i] = s[i];
      }
      s = temp;
   }


}
