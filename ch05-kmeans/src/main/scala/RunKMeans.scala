import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Random

object RunKMeans {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().getOrCreate()

    val data = spark.read.
      option("inferSchema", value = true).
      option("header", value = false).
      csv("hdfs://localhost:9000/aas/ch05/ds/kddcup.data").
      toDF(
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label")

    data.cache()

    val runKMeans = new RunKMeans(spark)

    // runKMeans.clusteringTake0(data)
    // runKMeans.clusteringTake1(data)
    // runKMeans.clusteringTake2(data)
    // runKMeans.clusteringTake3(data)
    // runKMeans.clusteringTake4(data)
    runKMeans.buildAnomalyDetector(data)

    data.unpersist()
  }

}

class RunKMeans(private val spark: SparkSession) {

  import spark.implicits._

  // Clustering, Take 0

  def clusteringTake0(data: DataFrame): Unit = {

    data.select("label").groupBy("label").count().orderBy($"count".desc).show(25)

    // non-numeric field 없애기
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()

    val assembler = new VectorAssembler().
      setInputCols(numericOnly.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
    val pipelineModel = pipeline.fit(numericOnly)
    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

    // cluster 출력해보기
    kmeansModel.clusterCenters.foreach(println)

    // 각 label 이 어느 cluster 에 속했는지 출력해보기
    val withCluster = pipelineModel.transform(numericOnly)

    withCluster.select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy($"cluster", $"count".desc).
      show(25)

    numericOnly.unpersist()
  }

  // Clustering, Take 1

  def clusteringScore0(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector")

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

  def clusteringScore1(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    // hyperparameter 추가 설정
    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("featureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, kmeans))

    val kmeansModel = pipeline.fit(data).stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

  def clusteringTake1(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    (20 to 100 by 20).map(k => (k, clusteringScore0(numericOnly, k))).foreach(println)
    (20 to 100 by 20).map(k => (k, clusteringScore1(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  // Clustering, Take 2

  def clusteringScore2(data: DataFrame, k: Int): Double = {
    val assembler = new VectorAssembler().
      setInputCols(data.columns.filter(_ != "label")).
      setOutputCol("featureVector")

    // scaler 추가
    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(Array(assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

  def clusteringTake2(data: DataFrame): Unit = {
    val numericOnly = data.drop("protocol_type", "service", "flag").cache()
    (60 to 270 by 30).map(k => (k, clusteringScore2(numericOnly, k))).foreach(println)
    numericOnly.unpersist()
  }

  // Clustering, Take 3

  /**
   * inputCol 에 들어오는 하나의 column 에 대해 one-hot encoding
   */
  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")
    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }

  def clusteringScore3(data: DataFrame, k: Int): Double = {
    // one-hot encoder 추가
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    val pipelineModel = pipeline.fit(data)

    val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    kmeansModel.summary.trainingCost
  }

  def clusteringTake3(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore3(data, k))).foreach(println)
  }

  // Clustering, Take 4

  /**
   * 자체 entropy 함수 정의
   */
  def entropy(counts: Iterable[Int]): Double = {
    val values = counts.filter(_ > 0)
    val n = values.map(_.toDouble).sum
    values.map { v =>
      val p = v / n
      -p * math.log(p)
    }.sum
  }

  def fitPipeline4(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    pipeline.fit(data)
  }

  def clusteringScore4(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline4(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data).
      select("cluster", "label").as[(Int, String)]
    val weightedClusterEntropy = clusterLabel.
      // Extract collections of labels, per cluster
      groupByKey { case (cluster, _) => cluster }.
      mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
      }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
  }

  def clusteringTake4(data: DataFrame): Unit = {
    (60 to 270 by 30).map(k => (k, clusteringScore4(data, k))).foreach(println)

    val pipelineModel = fitPipeline4(data, 180)
    val countByClusterLabel = pipelineModel.transform(data).
      select("cluster", "label").
      groupBy("cluster", "label").count().
      orderBy("cluster", "label")
    countByClusterLabel.show()
  }

  // Detect anomalies

  def buildAnomalyDetector(data: DataFrame): Unit = {
    val pipelineModel = fitPipeline4(data, 180)

    val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
    val centroids = kMeansModel.clusterCenters

    // data 를 pipeline 에 넣고 결과를 얻음
    val clustered = pipelineModel.transform(data)

    // 100 번째로 centroid 에서 거리가 먼 경우의 거리를 threshold 로 잡음
    val threshold = clustered.
      select("cluster", "scaledFeatureVector").as[(Int, Vector)].
      map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.
      orderBy($"value".desc).take(100).last

    // 이상치 출력해보기
    val originalCols = data.columns
    val anomalies = clustered.filter { row =>
      val cluster = row.getAs[Int]("cluster")
      val vec = row.getAs[Vector]("scaledFeatureVector")
      Vectors.sqdist(centroids(cluster), vec) >= threshold
    }.select(originalCols.head, originalCols.tail:_*)

    println(anomalies.first())
  }

}