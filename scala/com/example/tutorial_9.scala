package com.example

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry.{EuclideanVector, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.statisticalmodel.{LowRankGaussianProcess, MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_9 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/bfm.h5")).get

    val modelGroup = ui.createGroup("modelGroup")
    val ssmView = ui.show(modelGroup, model, "model")
    val idNoseTip = PointId(8156)
    val noseTipReference = model.referenceMesh.pointSet.point(idNoseTip)
    val noseTipMean = model.mean.pointSet.point(idNoseTip)
    val noseTipDeformation = (noseTipMean - noseTipReference) * 2.0
    val noseTipDomain = UnstructuredPointsDomain(IndexedSeq(noseTipReference))
    val noseTipDeformationAsSeq = IndexedSeq(noseTipDeformation)
    val noseTipDeformationField = DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](noseTipDomain, noseTipDeformationAsSeq)

    val observationGroup = ui.createGroup("observation")
    ui.show(observationGroup, noseTipDeformationField, "noseTip")
    val noise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3))
    val regressionData = IndexedSeq((noseTipReference, noseTipDeformation, noise))
    val gp : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = model.gp.interpolate(NearestNeighborInterpolator())

    val posteriorGP : LowRankGaussianProcess[_3D, EuclideanVector[_3D]] = LowRankGaussianProcess.regression(gp, regressionData)
    val posteriorSample : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]
    = posteriorGP.sampleAtPoints(model.referenceMesh.pointSet)
    val posteriorSampleGroup = ui.createGroup("posteriorSamples")
    for (i <- 0 until 1) {
      ui.show(posteriorSampleGroup, posteriorSample, "posteriorSample")
    }
    val littleNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 0.01)

    val pointOnLargeNose = noseTipReference + noseTipDeformation
    val discreteTrainingData = IndexedSeq((PointId(8156), pointOnLargeNose, littleNoise))
    val meshModelPosterior : StatisticalMeshModel = model.posterior(discreteTrainingData)
    val posteriorModelGroup = ui.createGroup("posteriorModel")
    ui.show(posteriorModelGroup, meshModelPosterior, "NoseyModel")
    val largeNoise = MultivariateNormalDistribution(DenseVector.zeros[Double](3), DenseMatrix.eye[Double](3) * 10.0)
    val discreteTrainingDataLargeNoise = IndexedSeq((PointId(8156), pointOnLargeNose, largeNoise))
    val discretePosteriorLargeNoise = model.posterior(discreteTrainingDataLargeNoise)
    val posteriorGroupLargeNoise = ui.createGroup("posteriorLargeNoise")
    ui.show(posteriorGroupLargeNoise, discretePosteriorLargeNoise, "NoisyNoseyModel")
  }
}
