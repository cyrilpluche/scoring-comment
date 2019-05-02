package helpers

import scala.annotation.tailrec
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql._
import org.apache.spark.sql.types._

object DataCleaner {

  val schemaStruct: StructType = StructType(
    StructField("id", IntegerType, true) ::
      StructField("target", DoubleType, true) ::
      StructField("comment_text", StringType, true) ::
      StructField("severe_toxicity", DoubleType, true) ::
      StructField("obscene", DoubleType, true) ::
      StructField("identity_attack", DoubleType, true) ::
      StructField("insult", DoubleType, true) ::
      StructField("threat", DoubleType, true) ::
      StructField("asian", DoubleType, true) ::
      StructField("atheist", DoubleType, true) ::
      StructField("bisexual", DoubleType, true) ::
      StructField("black", DoubleType, true) ::
      StructField("buddhist", DoubleType, true) ::
      StructField("christian", DoubleType, true) ::
      StructField("female", DoubleType, true) ::
      StructField("heterosexual", DoubleType, true) ::
      StructField("hindu", DoubleType, true) ::
      StructField("homosexual_gay_or_lesbian", DoubleType, true) ::
      StructField("intellectual_or_learning_disability", DoubleType, true) ::
      StructField("jewish", DoubleType, true) ::
      StructField("latino", DoubleType, true) ::
      StructField("male", DoubleType, true) ::
      StructField("muslim", DoubleType, true) ::
      StructField("other_disability", DoubleType, true) ::
      StructField("other_gender", DoubleType, true) ::
      StructField("other_race_or_ethnicity", DoubleType, true) ::
      StructField("other_religion", DoubleType, true) ::
      StructField("other_sexual_orientation", DoubleType, true) ::
      StructField("physical_disability", DoubleType, true) ::
      StructField("psychiatric_or_mental_illness", DoubleType, true) ::
      StructField("transgender", DoubleType, true) ::
      StructField("white", DoubleType, true) ::
      StructField("created_date", DateType, true) ::
      StructField("publication_id", IntegerType, true) ::
      StructField("parent_id", DoubleType, true) ::
      StructField("article_id", DoubleType, true) ::
      StructField("rating", StringType, true) ::
      StructField("funny", DoubleType, true) ::
      StructField("wow", DoubleType, true) ::
      StructField("sad", DoubleType, true) ::
      StructField("likes", DoubleType, true) ::
      StructField("disagree", DoubleType, true) ::
      StructField("sexual_explicit", DoubleType, true) ::
      StructField("identity_annotator_count", DoubleType, true) ::
      StructField("toxicity_annotator_count", DoubleType, true) :: Nil
  )

  def clean(initialDf: DataFrame, nb: Int): DataFrame = {
    // We drop column we don't want
    val dfAfterDrop = initialDf
      .drop("id")
      .drop("parent_id")
      .drop("article_id")
      .drop("publication_id")
      .drop("transgender")
      .drop("other_gender")
      .drop("heterosexual")
      .drop("bisexual")
      .drop("other_sexual_orientation")
      .drop("hindu")
      .drop("buddhist")
      .drop("atheist")
      .drop("other_religion")
      .drop("asian")
      .drop("latino")
      .drop("other_race_or_ethnicity")
      .drop("physical_disability")
      .drop("intellectual_or_learning_disability")
      .drop("other_disability")
      .drop("created_date")

    val converter = udf(convertTarget)
    val dfTargetConverted: DataFrame = dfAfterDrop.withColumn("target", converter(dfAfterDrop("target")))

    val dfWithoutNull = removeNullValues(dfTargetConverted, 0)

    try {
      val limit: Int = nb.toInt
      dfWithoutNull
        .filter(dfWithoutNull("target") >= 0 && dfWithoutNull("target") <= 1)
        .limit(limit)
    } finally {
      dfWithoutNull
        .filter(dfWithoutNull("target") >= 0 && dfWithoutNull("target") <= 1)
        .limit(10000)
    }
  }

  // Set target columns to binary values 1.0 or 0.0
  def convertTarget: (Double => Double) = { x => {
    if (x >= 0.5 ) 1.0
    else 0.0
  }}

  def cleanAfterIndex(df: DataFrame): DataFrame = {
    val finalDf: DataFrame = df.select("target", "scaledFeatures")
    finalDf
  }

  // Replace null values by 0.0 for numerical columns
  // Replace null values by "" for String columns
  @tailrec
  def removeNullValues(df: DataFrame, i: Int): DataFrame = {
    if (i == df.columns.length) {
      df
    } else {
      if (!(Array("target", "comment_text", "rating") contains df.schema.fields(i).name)) {
        // Numerical column
        val dfCopy: DataFrame = df.na.fill(0.0, Array(df.schema.fields(i).name))
        removeNullValues(dfCopy, i + 1)
      } else if (df.schema.fields(i).name == "rating") {
        // String column
        val dfCopy: DataFrame = df.na.fill("", Array(df.schema.fields(i).name))
        removeNullValues(dfCopy, i + 1)
      } else {
        removeNullValues(df, i + 1)
      }
    }
  }

  def splitData(df: DataFrame, mode: Int): Array[DataFrame] = {
    if (mode == 3) {
      val Array(train, test) = df.randomSplit(Array(0.7, 0.3))
      Array(train, test)
    } else {
      Array(df, df)
    }

  }

}
