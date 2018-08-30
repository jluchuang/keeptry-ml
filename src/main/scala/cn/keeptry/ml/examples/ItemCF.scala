package cn.keeptry.ml.examples

import cn.keeptry.ml.knns.SymmetricAlgo
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object ItemCF {

    val RATING_PATH = "file:///Users/chuang/Downloads/ml-100k/u.data"
    val ITEM_PATH = "file:///Users/chuang/Downloads/ml-100k/u.item"

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().master("local").appName("RunRecommender").config("spark.sql.crossJoin.enabled", "true").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        spark.sparkContext.setCheckpointDir("hdfs://localhost:8020/tmp/")

        val base = "hdfs://localhost:8020/knns/"

        val ratingData = spark.sparkContext.textFile(RATING_PATH).map(l => {
            val rateInfo = l.split("\t")
            (rateInfo(0), (rateInfo(1).toInt, rateInfo(2).toDouble))
        })

        val dataSplit = ratingData.randomSplit(Array(0.9, 0.1), 2018)
        SymmetricAlgo.evaluate(spark.sparkContext, dataSplit(0), dataSplit(1))

        val itemInfo = spark.sparkContext.textFile(ITEM_PATH).map(l => {
            val firstIndex = l.indexOf("|")
            val itemId = l.substring(0, firstIndex).toInt

            val secondIndex = l.indexOf("|", firstIndex + 1)
            val itemName = l.substring(firstIndex + 1, secondIndex)

            (itemId, itemName)
        }).collectAsMap()

        val bItemInfo = spark.sparkContext.broadcast(itemInfo)

        val itemSim: RDD[(Int, Array[(Int, Double)])] = SymmetricAlgo.calculateItemSim(spark.sparkContext, ratingData)
        val predictResult: RDD[(String, Array[(Int, Double, String)])] =
            SymmetricAlgo.predict(spark.sparkContext, ratingData, itemSim, 10)

        predictResult.map {
            case (userId, items) =>
                val pre = items.map {
                    case (itemId, score, reason) => s"${itemId}|${bItemInfo.value.getOrElse(itemId, "")}|${score}|${reason}"
                }.mkString("/")

                s"${userId}\t${pre}"
        }.saveAsTextFile("file:///Users/chuang/Downloads/ml-100k/predict_result")

    }
}
