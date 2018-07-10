package com.xiaomi.teg.demo.music.als

import com.xiaomi.teg.demo.music.als.Audioscrobbler.BASE_PATH
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.Map
import scala.util.Random

object Audioscrobbler {

    val BASE_PATH = "file:///Users/chuang/xiaomi-src/angelia/profiledata_06-May-2005"

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder().master("local").appName("RunRecommender").config("spark.sql.crossJoin.enabled", "true").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        spark.sparkContext.setCheckpointDir("hdfs://localhost:8020/tmp/")

        val base = "hdfs://localhost:8020/profiledata_06-May-2005/"

        val rawUserArtistData = spark.read.textFile(BASE_PATH + "/" + "user_artist_data.txt")
        val rawArtistData = spark.read.textFile(BASE_PATH + "/" + "artist_data.txt")
        val rawArtistAlias = spark.read.textFile(BASE_PATH + "/" + "artist_alias.txt")

        val runRecommender = new RunRecommender(spark)
        runRecommender.model(rawUserArtistData, rawArtistData, rawArtistAlias)
    }
}

class RunRecommender(private val spark: SparkSession) {

    import spark.implicits._

    def prepration(rawUserArtistData: Dataset[String],
                   rawArtistData: Dataset[String],
                   rawArtistAlias: Dataset[String]): Unit = {
        rawUserArtistData.take(5).foreach(println)

        val userArtistDF = rawUserArtistData.map(line => {
            val Array(user, artist, _*) = line.split(' ')
            (user.toInt, artist.toInt)
        }).toDF("user", "artist")

        userArtistDF.agg(min("user"), max("user"),
            min("artist"), max("artist")).show()

        val artistByID = buildArtistByID(rawArtistData)
        val artistAlias = buildArtistByAlias(rawArtistAlias)

        val (badID, goodID) = artistAlias.head
        //        artistByID.filter($"id".isin(goodID, badID)).show
    }

    def buildArtistByID(rawArtistData: Dataset[String]): DataFrame = {
        rawArtistData.flatMap { line =>
            val (id, name) = line.span(_ != '\t')
            if (name.isEmpty) {
                None
            }
            else {
                try {
                    Some((id.toInt, name.trim))
                }
                catch {
                    case _: NumberFormatException => None
                }
            }
        }.toDF("id", "name")
    }

    def buildArtistByAlias(rawArtistAlias: Dataset[String]): Map[Int, Int] = {
        rawArtistAlias.flatMap { line =>
            val Array(artist, alias) = line.split('\t')
            if (artist.isEmpty) {
                None
            }
            else {
                Some(artist.toInt, alias.toInt)
            }
        }.collect().toMap
    }

    def buildCounts(rawUserAritstData: Dataset[String],
                    bArtistAlias: Broadcast[Map[Int, Int]]): DataFrame = {
        rawUserAritstData.map { line =>
            val Array(userID, artistID, count) = line.split(' ').map(_.toInt)
            val finalArtistID = bArtistAlias.value.getOrElse(artistID, artistID)
            (userID, finalArtistID, count)
        }.toDF("user", "artist", "count")
    }

    def model(rawUserArtistData: Dataset[String],
              rawArtistData: Dataset[String],
              rawArtistAlias: Dataset[String]): Unit = {
        val bArtistAlias = spark.sparkContext.broadcast(buildArtistByAlias(rawArtistAlias))
        val trainData = buildCounts(rawUserArtistData, bArtistAlias).cache()

        /*
        setImplicitPrefs  是否采用隐式偏好
        setRank           用于矩阵分解的等级（正）
        setRegParam       正则化参数（非负）
        setAlpha          参数为隐式偏好公式中的alpha参数（非负）
        setMaxIter        最大迭代次数
        */
        val model = new ALS().
                setSeed(Random.nextLong()).
                setImplicitPrefs(true).
                setRank(10).
                setRegParam(0.01).
                setAlpha(1.0).
                setMaxIter(20).
                setUserCol("user").
                setItemCol("artist").
                setRatingCol("count").
                setPredictionCol("prediction").
                fit(trainData)
        trainData.unpersist()

        model.save(BASE_PATH + "/" + "model")

        model.userFactors.select("features").show(truncate = false)
        model.itemFactors.select("features").show(truncate = false)

        val userID = 2093760

        val existingArtistIDs = trainData.
                filter(col("user") === userID).
                select("artist").as[Int].collect()

        val artistByID = buildArtistByID(rawArtistData)

        artistByID.filter($"id" isin (existingArtistIDs: _*)).show()
        val topRecommendations = makeRecommendations(model, userID, 5)
        topRecommendations.show()

        val recommendArtistIDs = topRecommendations.select("artist").as[Int].collect()
        artistByID.filter($"id" isin (recommendArtistIDs: _*)).show()
        model.userFactors.unpersist()
        model.itemFactors.unpersist()
    }

    def makeRecommendations(model: ALSModel, userID: Int, howMany: Int): DataFrame = {
        val toRecommend = model.itemFactors
                .select($"id".as("artist"))
                .withColumn("user", lit(userID))
        model.transform(toRecommend)
                .select("artist", "prediction")
                .orderBy($"prediction".desc)
                .limit(howMany)
    }
}

