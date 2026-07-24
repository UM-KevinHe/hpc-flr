import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel
import scala.collection.mutable
import breeze.linalg._
import breeze.numerics._

object OneShot {

  final case class Result(
      beta: Map[String, Double],
      gamma: Map[Int, Double],
      seconds: Double
  )

  private final class GroupPartitioner(assignment: Map[Int, Int], parts: Int) extends Partitioner {
    def numPartitions: Int = parts
    def getPartition(key: Any): Int = assignment.getOrElse(key.asInstanceOf[Int], 0)
  }

  private def assignGroups(counts: Array[(Int, Int)], parts: Int): Map[Int, Int] = {
    val loads = Array.fill(parts)(0L)
    val out = mutable.Map[Int, Int]()
    for ((id, cnt) <- counts.sortBy(-_._2)) {
      var best = 0
      var min = Long.MaxValue
      var k = 0
      while (k < parts) {
        if (loads(k) < min) { min = loads(k); best = k }
        k += 1
      }
      out(id) = best
      loads(best) += cnt
    }
    out.toMap
  }

  def partitionByGroup(
      sc: SparkContext,
      rows: RDD[(Int, Double, Double, Array[Double])],
      parts: Int
  ): RDD[(Int, Double, Double, Array[Double])] = {
    val counts = rows.map(r => (r._1, 1)).reduceByKey(_ + _).collect()
    val assignment = sc.broadcast(assignGroups(counts, parts))
    rows
      .map(r => (r._1, r))
      .partitionBy(new GroupPartitioner(assignment.value, parts))
      .map(_._2)
  }

  private def buildBlocks(
      data: Array[(Int, Double, Double, Array[Double])],
      nbeta: Int
  ): (DenseMatrix[Double], DenseVector[Double], DenseVector[Double], Array[Int], Array[Int]) = {
    val nrow = data.length
    val Y = DenseVector(data.map(_._2))
    val N = DenseVector(data.map(_._3))
    val xData = new Array[Double](nrow * nbeta)
    var r = 0
    while (r < nrow) {
      val feats = data(r)._4
      var c = 0
      while (c < nbeta) { xData(r + c * nrow) = feats(c); c += 1 }
      r += 1
    }
    val X = new DenseMatrix[Double](nrow, nbeta, xData)
    val ids = data.map(_._1)
    val starts = mutable.ArrayBuffer[Int]()
    val ordered = mutable.ArrayBuffer[Int]()
    var s = 0
    while (s < nrow) {
      val cur = ids(s)
      var e = s + 1
      while (e < nrow && ids(e) == cur) e += 1
      starts += s
      ordered += cur
      s = e
    }
    starts += nrow
    (X, Y, N, starts.toArray, ordered.toArray)
  }

  def fitPartition(
      rows: Iterator[(Int, Double, Double, Array[Double])],
      nbeta: Int,
      gammaInit: Double,
      tol: Double,
      maxIter: Int,
      bound: Double
  ): Iterator[(DenseMatrix[Double], DenseVector[Double])] = {
    val data = rows.toArray.sortBy(_._1)
    if (data.isEmpty) return Iterator.empty

    val (x, y, n, bounds, ordered) = buildBlocks(data, nbeta)
    val nrow = x.rows
    val mk = ordered.length
    val gamma = Array.fill(mk)(gammaInit)
    var beta = DenseVector.zeros[Double](nbeta)

    def gammaObs(): DenseVector[Double] = {
      val arr = new Array[Double](nrow)
      var b = 0
      while (b < mk) { java.util.Arrays.fill(arr, bounds(b), bounds(b + 1), gamma(b)); b += 1 }
      DenseVector(arr)
    }

    var iter = 0
    var crit = Double.MaxValue
    while (iter < maxIter && crit > tol) {
      val prob = sigmoid(gammaObs() + x * beta)
      val pq = (n *:* prob *:* (1.0 - prob)).map(v => if (v <= 0.0) 1e-10 else v)
      val residual = y - (n *:* prob)

      val infoGamma = DenseVector.zeros[Double](mk)
      val infoBetaGamma = DenseMatrix.zeros[Double](nbeta, mk)
      var b = 0
      while (b < mk) {
        val lo = bounds(b)
        val hi = bounds(b + 1)
        val w = pq(lo until hi).copy
        infoGamma(b) = sum(w)
        infoBetaGamma(::, b) := (x(lo until hi, ::).copy.t * w)
        b += 1
      }
      val infoGammaInv = infoGamma.map(1.0 / _)
      val infoBeta = x.t * (x(::, *) *:* pq)

      val j1 = DenseMatrix.zeros[Double](nbeta, mk)
      b = 0
      while (b < mk) { j1(::, b) := infoBetaGamma(::, b) * infoGammaInv(b); b += 1 }
      val schur = infoBeta - j1 * infoBetaGamma.t
      val sInv = pinv(schur)
      val j2 = sInv * j1

      val firstTerm = infoGammaInv + diag(j1.t * sInv * j1)
      val h1 = DenseVector.zeros[Double](nrow)
      val h2 = DenseVector.zeros[Double](nrow)
      b = 0
      while (b < mk) {
        val lo = bounds(b)
        val hi = bounds(b + 1)
        h1(lo until hi) := DenseVector.fill(hi - lo)(firstTerm(b))
        h2(lo until hi) := (-x(lo until hi, ::) * j2(::, b)).toDenseVector
        b += 1
      }
      val h3 = sum(x *:* (x * sInv), Axis._1)
      val h = pq *:* (h1 + h2 * 2.0 + h3)
      val ypa = residual + (h *:* (0.5 - prob))

      val uGamma = DenseVector.zeros[Double](mk)
      b = 0
      while (b < mk) { uGamma(b) = sum(ypa(bounds(b) until bounds(b + 1))); b += 1 }
      val uBeta = x.t * ypa

      val dGamma = (infoGammaInv *:* uGamma) + j2.t * (j1 * uGamma - uBeta)
      val dBeta = sInv * uBeta - j2 * uGamma

      b = 0
      while (b < mk) { gamma(b) = math.max(math.min(gamma(b) + dGamma(b), bound), -bound); b += 1 }
      beta = beta + dBeta
      crit = norm(dBeta, Double.PositiveInfinity)
      iter += 1
    }

    val prob = sigmoid(gammaObs() + x * beta)
    val pq = (n *:* prob *:* (1.0 - prob)).map(v => if (v <= 0.0) 1e-10 else v)
    val infoGamma = DenseVector.zeros[Double](mk)
    val infoBetaGamma = DenseMatrix.zeros[Double](nbeta, mk)
    var b = 0
    while (b < mk) {
      val lo = bounds(b)
      val hi = bounds(b + 1)
      val w = pq(lo until hi).copy
      infoGamma(b) = sum(w)
      infoBetaGamma(::, b) := (x(lo until hi, ::).copy.t * w)
      b += 1
    }
    val infoGammaInv = infoGamma.map(1.0 / _)
    val infoBeta = x.t * (x(::, *) *:* pq)
    val omega = infoBeta - (infoBetaGamma(*, ::) *:* infoGammaInv) * infoBetaGamma.t
    Iterator((omega, omega * beta))
  }

  def refitGammaPartition(
      rows: Iterator[(Int, Double, Double, Array[Double])],
      nbeta: Int,
      gammaInit: Double,
      betaHat: DenseVector[Double],
      tol: Double,
      maxIter: Int,
      bound: Double
  ): Iterator[(Int, Double)] = {
    val data = rows.toArray.sortBy(_._1)
    if (data.isEmpty) return Iterator.empty

    val (x, y, n, bounds, ordered) = buildBlocks(data, nbeta)
    val nrow = x.rows
    val mk = ordered.length
    val offset = x * betaHat
    val gamma = Array.fill(mk)(gammaInit)

    def gammaObs(): DenseVector[Double] = {
      val arr = new Array[Double](nrow)
      var b = 0
      while (b < mk) { java.util.Arrays.fill(arr, bounds(b), bounds(b + 1), gamma(b)); b += 1 }
      DenseVector(arr)
    }

    var iter = 0
    var crit = Double.MaxValue
    while (iter < maxIter && crit > tol) {
      val prob = sigmoid(gammaObs() + offset)
      val pq = (n *:* prob *:* (1.0 - prob)).map(v => if (v <= 0.0) 1e-10 else v)
      val residual = y - (n *:* prob)

      val infoGamma = DenseVector.zeros[Double](mk)
      val infoBetaGamma = DenseMatrix.zeros[Double](nbeta, mk)
      var b = 0
      while (b < mk) {
        val lo = bounds(b)
        val hi = bounds(b + 1)
        val w = pq(lo until hi).copy
        infoGamma(b) = sum(w)
        infoBetaGamma(::, b) := (x(lo until hi, ::).copy.t * w)
        b += 1
      }
      val infoGammaInv = infoGamma.map(1.0 / _)
      val infoBeta = x.t * (x(::, *) *:* pq)
      val j1 = DenseMatrix.zeros[Double](nbeta, mk)
      b = 0
      while (b < mk) { j1(::, b) := infoBetaGamma(::, b) * infoGammaInv(b); b += 1 }
      val schur = infoBeta - j1 * infoBetaGamma.t
      val sInv = pinv(schur)

      val firstTerm = infoGammaInv + diag(j1.t * sInv * j1)
      val h1 = DenseVector.zeros[Double](nrow)
      val h2 = DenseVector.zeros[Double](nrow)
      b = 0
      while (b < mk) {
        val lo = bounds(b)
        val hi = bounds(b + 1)
        h1(lo until hi) := DenseVector.fill(hi - lo)(firstTerm(b))
        h2(lo until hi) := (-x(lo until hi, ::) * (sInv * j1(::, b))).toDenseVector
        b += 1
      }
      val h3 = sum(x *:* (x * sInv), Axis._1)
      val h = pq *:* (h1 + h2 * 2.0 + h3)
      val ypa = residual + (h *:* (0.5 - prob))

      var maxStep = 0.0
      b = 0
      while (b < mk) {
        val dGamma = infoGammaInv(b) * sum(ypa(bounds(b) until bounds(b + 1)))
        val updated = math.max(math.min(gamma(b) + dGamma, bound), -bound)
        maxStep = math.max(maxStep, math.abs(updated - gamma(b)))
        gamma(b) = updated
        b += 1
      }
      crit = maxStep
      iter += 1
    }

    ordered.zip(gamma).iterator
  }

  def run(
      spark: SparkSession,
      df: DataFrame,
      successCol: String,
      trialsCol: String,
      groupCol: String,
      numPartitions: Int,
      excludedCols: Set[String] = Set(),
      tol: Double = 1e-6,
      maxIter: Int = 100,
      bound: Double = 10.0
  ): Result = {
    import spark.implicits._
    val sc = spark.sparkContext
    val excluded = excludedCols ++ Set(successCol, trialsCol, groupCol)
    val featureCols = df.columns.filterNot(excluded.contains)
    val nbeta = featureCols.length

    val selectExprs = Seq(
      s"cast($groupCol as int) as group_id",
      s"cast($successCol as double) as successes",
      s"cast($trialsCol as double) as trials"
    ) ++ featureCols.map(c => s"coalesce(cast($c as double), 0.0) as $c")
    val clean = df.selectExpr(selectExprs: _*)

    val (sumSuccess, sumTrials) = clean.selectExpr("sum(successes)", "sum(trials)").as[(Double, Double)].first()
    val pBar = sumSuccess / sumTrials
    val gammaInit = math.log(pBar / (1.0 - pBar))

    val rows: RDD[(Int, Double, Double, Array[Double])] = clean.rdd.map { row =>
      val id = row.getInt(0)
      val y = row.getDouble(1)
      val n = row.getDouble(2)
      val feats = new Array[Double](nbeta)
      var i = 0
      while (i < nbeta) { feats(i) = row.getDouble(i + 3); i += 1 }
      (id, y, n, feats)
    }

    val partitioned = partitionByGroup(sc, rows, numPartitions).persist(StorageLevel.MEMORY_AND_DISK_SER)
    partitioned.count()

    val startTime = System.nanoTime()

    val zero = (DenseMatrix.zeros[Double](nbeta, nbeta), DenseVector.zeros[Double](nbeta))
    val (sumOmega, sumNumer) = partitioned
      .mapPartitions(part => fitPartition(part, nbeta, gammaInit, tol, maxIter, bound))
      .aggregate(zero)(
        (acc, v) => (acc._1 + v._1, acc._2 + v._2),
        (a, b) => (a._1 + b._1, a._2 + b._2)
      )
    val betaHat = pinv(sumOmega) * sumNumer

    val bBeta = sc.broadcast(betaHat)
    val gammaPairs = partitioned
      .mapPartitions(part => refitGammaPartition(part, nbeta, gammaInit, bBeta.value, tol, maxIter, bound))
      .collect()
    bBeta.destroy()
    partitioned.unpersist()

    val seconds = (System.nanoTime() - startTime) / 1e9
    Result(
      featureCols.zip(betaHat.toArray).toMap,
      gammaPairs.toMap,
      seconds
    )
  }
}
