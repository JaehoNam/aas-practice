import org.apache.spark.sql.{DataFrame, SparkSession}

case class MatchData(
  id_1: Int,
  id_2: Int,
  cmp_fname_c1: Option[Double],
  cmp_fname_c2: Option[Double],
  cmp_lname_c1: Option[Double],
  cmp_lname_c2: Option[Double],
  cmp_sex: Option[Int],
  cmp_bd: Option[Int],
  cmp_bm: Option[Int],
  cmp_by: Option[Int],
  cmp_plz: Option[Int],
  is_match: Boolean
)

object RunIntro extends Serializable {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Intro")
      .getOrCreate
    import spark.implicits._

    val parsed = spark.read
      .option("header", "true")
      .option("nullValue", "?")
      .option("inferSchema", "true")
      .csv("hdfs://localhost:9000/aas/ch02/ds")

    parsed.cache()

    val matchData = parsed.as[MatchData]
    val scored = matchData.map { md =>
      (scoreMatchData(md), md.is_match)
    }.toDF("score", "is_match")
    crossTabs(scored, 4.5).show()
    crossTabs(scored, 4.0).show()
    crossTabs(scored, 3.5).show()
    crossTabs(scored, 3.0).show()
    crossTabs(scored, 2.5).show()
    crossTabs(scored, 2.0).show()
    crossTabs(scored, 1.5).show()
  }

  def crossTabs(scored: DataFrame, t: Double): DataFrame = {
    scored.
      selectExpr(s"score >= $t as above", "is_match").
      groupBy("above").
      pivot("is_match", Seq("true", "false")).
      count()
  }

  case class Score(value: Double) {
    def +(oi: Option[Int]) = {
      Score(value + oi.getOrElse(0))
    }
  }

  def scoreMatchData(md: MatchData): Double = {
    (Score(md.cmp_lname_c1.getOrElse(0.0)) + md.cmp_plz +
      md.cmp_by + md.cmp_bd + md.cmp_bm).value
  }
}