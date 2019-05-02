package helpers

import org.apache.spark.sql.DataFrame

object Evaluation {
  def calculateMetrics(df: DataFrame): Unit = {

    val counttotal = df.count()
    println(s"Row handled         : $counttotal")

    val correct = df.filter(df.col("target") === df.col("prediction")).count()
    val ratioCorrect = correct.toDouble / counttotal.toDouble
    println(s"Correct predictions : $correct | $ratioCorrect")

    val wrong = df.filter(df.col("target") =!= df.col("prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    println(s"Correct predictions : $wrong | $ratioWrong")

    val truep = df.filter(df.col("prediction") === 1.0).filter(df.col("target") === df.col("prediction")).count()
    println(s"True positive       : $truep")
    val truen = df.filter(df.col("prediction") === 1.0).filter(df.col("target") =!= df.col("prediction")).count()
    println(s"True negative       :  $truen")
    val falsen = df.filter(df.col("prediction") === 0.0).filter(df.col("target") =!= df.col("prediction")).count()
    println(s"False negative      : $falsen")
    val falsep = df.filter(df.col("prediction") === 0.0).filter(df.col("target") === df.col("prediction")).count()
    println(s"False positive      : $falsep")

    val precision = truep / (truep + truen)
    println(s"Precision           : $precision")
    val recall = truep / (truep + falsen)
    println(s"Recall              : $precision")

  }
}
