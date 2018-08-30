package cn.keeptry.ml.knns

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import cn.keeptry.ml.utils.LogliLikehood.logLike
import org.apache.log4j.Logger
import org.apache.spark.mllib.rdd.MLPairRDDFunctions.fromPairRDD

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object SymmetricAlgo {

    val logger = Logger.getLogger(SymmetricAlgo.getClass)

    /**
      * Item CF ：基于item协同
      *
      * @param sc
      * @param data        : User -> item, rating
      * @param minFreq     : 共现次数阈值
      * @param topSimItems : 相似商品截断阈值
      * @return
      */
    def calculateItemSim(sc: SparkContext,
                         data: RDD[(String, (Int, Double))],
                         minFreq: Int = 1,
                         topSimItems: Int = 200): RDD[(Int, Array[(Int, Double)])] = {

        val userRatings = data.groupByKey()
        val itemCounts = data.map(l => (l._2._1.toInt, 1)).reduceByKey(_ + _)
        val itemCountsMap = sc.broadcast(itemCounts.collectAsMap())

        val all_event_num = sc.broadcast(userRatings.count())

        val itemSim: RDD[(Int, Array[(Int, Double)])] = userRatings
                .flatMap { l =>
                    val items = l._2.toList
                    val pairs = ListBuffer[((Int, Int), Long)]()
                    for (i <- items.indices; j <- i + 1 until items.length) {
                        var (itemi, itemj) = (items(i), items(j))
                        if (itemi._1 > itemj._1) {
                            itemi = items(j)
                            itemj = items(i)
                        }
                        pairs.append(((itemi._1, itemj._1), 1))
                    }
                    pairs
                }
                .reduceByKey(_ + _)
                // 同时听过 itemi 和 itemj 的人数（c[i][j]）共现矩阵
                .flatMap {
            case ((item1, item2), freq) =>
                val pairs = ListBuffer[(Int, (Int, Double))]()
                if (freq >= minFreq) {
                    // loglikelihood ratio
                    val a: Long = itemCountsMap.value(item1) // 喜欢item1的人数
                    val b: Long = itemCountsMap.value(item2) // 喜欢item2的人数

                    val sim = logLike(a, b, freq, all_event_num.value)
                    pairs.append((item1, (item2, sim)))
                    pairs.append((item2, (item1, sim)))
                }
                pairs
        }
                .topByKey(topSimItems)(Ordering[Double].on(_._2)) // 这里是不是可以卡一个相似度的阈值?
                .mapValues(
            arr => {
                // 归一化
                val maxSim = arr.map(_._2).max
                arr.map(x => (x._1, x._2 / maxSim))
            })

        itemSim
    }

    /**
      * 根据用户打分和item协同矩阵进行item推荐的分值预估
      *
      * @param sc
      * @param ratingData     : userId -> itemId, rating
      * @param itemSimRDD     : 协同item相似度矩阵
      * @param ratingThresold : 推荐召回截断阈值
      * @return
      */
    def predict(sc: SparkContext,
                ratingData: RDD[(String, (Int, Double))],
                itemSimRDD: RDD[(Int, Array[(Int, Double)])],
                ratingThreshold: Int): RDD[(String, Array[(Int, Double, String)])] = {

        val globalMean = ratingData.map(l => l._2._2).mean()

        val (bi, bu) = calculateBiBu(sc, ratingData)
        val biMap = sc.broadcast(bi)
        val buMap = sc.broadcast(bu)

        ratingData
                .map {
                    case (userId, (itemId, rating)) => (itemId, (userId, rating))
                }
                .join(itemSimRDD)
                .flatMap {
                    case (itemId, ((userId, rating), simItems)) => {
                        simItems.map {

                            case (simId, simValue) => {
                                val buj = globalMean + biMap.value.getOrElse(itemId, 0D) + buMap.value.getOrElse(userId, 0D)
                                ((userId, simId), (simValue * (rating - buj), simValue, itemId.toString))
                            }
                        }
                    }
                }
                .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2, s"${a._3},${b._3}"))
                .map {
                    case ((userId, itemId), (sumRate, simSum, reason)) => {
                        val bui = globalMean + biMap.value.getOrElse(itemId, 0D) + buMap.value.getOrElse(userId, 0D)
                        val s = if (simSum == 0) 0 else sumRate / simSum
                        val preRate = math.max(math.min(s + bui, 5), 0)
                        (userId, (itemId, preRate, reason))
                    }
                }
                .topByKey(ratingThreshold)(Ordering[Double].on(_._2))

    }

    /**
      *
      * @param sc
      * @param trainData
      * @param testData
      */
    def evaluate(sc: SparkContext,
                 trainData: RDD[(String, (Int, Double))],
                 testData: RDD[(String, (Int, Double))]): Unit = {

        val itemSim = calculateItemSim(sc, trainData)

        val globalMean = trainData.map(l => l._2._2).mean()

        val (bi, bu) = calculateBiBu(sc, trainData)

        val biMap = sc.broadcast(bi)
        val buMap = sc.broadcast(bu)
        val testPairMap = sc.broadcast(testData.map(l => ((l._1, l._2._1), true)).collectAsMap())
        val testUserMap = sc.broadcast(testData.map(l => (l._1, true)).collectAsMap())
        val testItemMap = sc.broadcast(testData.map(l => (l._2._1, true)).collectAsMap())

        // itemId， (user, trueScore)
        // 训练数据中用户对某个商品的打分
        // 因为Predict只会使用到验证集中的用户
        val testUserHistory: RDD[(Int, (String, Double))] = trainData
                .filter(l => testUserMap.value.contains(l._1))
                .map(l => (l._2._1, (l._1, l._2._2)))

        // (item1, (item2, simValue))
        // 后面的指标计算只需要用到验证集里面包含的商品
        val similaritiesInTest: RDD[(Int, Array[(Int, Double)])] = itemSim.filter(l => testItemMap.value.contains(l._1))

        // 对验证集里面的item做打分
        val statistics: RDD[((String, Int), Double)] = testUserHistory
                .join(similaritiesInTest)
                .flatMap {
                    case (item, ((user, rating), simItemIterable)) =>
                        simItemIterable.map {
                            case (simItem, sim) => {
                                val buj = globalMean + biMap.value.getOrElse(item, 0D) + buMap.value.getOrElse(user, 0D)
                                ((user, simItem), (sim * (rating - buj), sim))
                            }
                        }
                }
                .filter {
                    case ((user, simItem), _) =>
                        testPairMap.value.contains((user, simItem))
                }
                .groupByKey(10)
                .mapValues(sims =>
                    sims.reduce((a, b) => (a._1 + b._1, a._2 + b._2))
                )
                // A basic collaborative filtering algorithm, taking into account the mean ratings of each user.
                // https://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic
                .map {
            case ((user, item), (ratingSum, simSum)) =>
                val bui = globalMean + biMap.value.getOrElse(item, 0D) + buMap.value.getOrElse(user, 0D)
                val s = if (simSum == 0) 0 else ratingSum / simSum
                val preRate = math.max(math.min(s + bui, 5), 0)
                (user, (item, preRate))
        }
                .topByKey(40)(Ordering[Double].on(_._2))
                .flatMap {
                    case (user, preItems) => {
                        preItems.map {
                            case (item, preRate) => {
                                ((user, item), preRate)
                            }
                        }
                    }
                }

        val trueScore: RDD[((String, Int), Double)] = testData.map(l => ((l._1, l._2._1), l._2._2))

        val err = statistics.join(trueScore).map(l => l._2).map(l => math.abs(l._1 - l._2))
        val rmse = math.sqrt(err.map(ll => ll * ll).mean())
        val mae = err.mean()

        logger.error(s"RMSE = $rmse")
        logger.error(s"MAE = $mae")
    }

    /**
      * implement https://github.com/NicolasHug/Surprise/blob/master/surprise/prediction_algorithms/optimize_baselines.pyx
      *
      * @param sc
      * @param userRating : userId -> (songId, rate)
      * @param lambda2
      * @param lambda3
      * @param epochs
      * @return
      */
    def calculateBiBu(sc: SparkContext,
                      userRating: RDD[(String, (Int, Double))],
                      lambda2: Int = 25,
                      lambda3: Int = 10,
                      epochs: Int = 20): (Map[Int, Double], Map[String, Double]) = {
        val globalMean = userRating.map {
            case (userId, (songId, rate)) => rate
        }.mean

        val u = sc.broadcast(globalMean)

        var bi = mutable.Map[Int, Double]()
        var bu = mutable.Map[String, Double]()

        for (dummy <- 0 until epochs) {
            bi ++= userRating
                    .map {
                        case (userId, (songId, rate)) => (songId, (userId, rate))
                    }
                    .groupByKey()
                    .map {
                        case (songId, ratings) => {
                            val sumI = ratings.map {
                                case (userId, rate) => {
                                    rate - u.value - bu.getOrElse(userId, 0D)
                                }
                            }.sum
                            val biValue = sumI / (lambda2 + ratings.size)
                            (songId, biValue)
                        }
                    }.collectAsMap()

            bu ++= userRating.groupByKey()
                    .map {
                        case (userId, ratings) => {

                            val sumU = ratings.map {
                                case (songId, rate) => {
                                    rate - u.value - bi.getOrElse(songId, 0D)
                                }
                            }.sum

                            val buValue = sumU / (lambda3 + ratings.size)
                            (userId, buValue)
                        }
                    }.collectAsMap()
        }

        (bi.toMap, bu.toMap)
    }
}
