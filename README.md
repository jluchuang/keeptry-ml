## keeptry-ml-demo 

keeptry machine learning demos 

### 环境

spark 2.3.1 
scala 2.11
hdfs 3.1.0
[环境搭建(待完善)](http://www.keeptry.cn)

### demos

- Spark ALS: 基于Audioscrobbler数据集的音乐推荐
- Spark ItemCF: 基于MoveLens数据集的电影推荐

### Benchmarks

#### Spark ItemCF
With datasets on MovieLens 100K and 1M. 

**MovieLens 100K** 

- RMSE = 0.9049792521615789
- MAE = 0.7078706510784817
 
**MovieLens 1M**

- RMSE = 0.8624739281707584
- MAE = 0.6485303967964491

**References**

1. [Surprise KNNBaseline](https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)
2. [Factor in the Neighbors: Scalable and Accurate Collaborative Filtering](http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf)
3. [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
