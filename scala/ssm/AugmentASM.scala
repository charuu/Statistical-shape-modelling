package ssm

import java.io.File

import scalismo.common.{Field, NearestNeighborInterpolator, RealSpace}
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{ActiveShapeModelIO, MeshIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, PDKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess}
import scalismo.ui.api.ScalismoUI

object AugmentASM {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    // val modelView = ui.show(modelGroup, model.statisticalModel, "shapeModel")
    val zeroMean = Field(RealSpace[_3D], (pt: Point[_3D]) => EuclideanVector(0, 0, 0))
    val scalarValuedGaussianKernel: PDKernel[_3D] = GaussianKernel(sigma = 100.0) * 10.0
    val scalarValuedGaussianKernel2: PDKernel[_3D] = GaussianKernel(sigma = 150.0) * 20.0
    val scalarValuedGaussianKernel3: PDKernel[_3D] = GaussianKernel(sigma = 100.0) * 10.0
    // val sym = symmetrizeKernel(scalarValuedGaussianKernel)

    val augmentedCov = DiagonalKernel(scalarValuedGaussianKernel, scalarValuedGaussianKernel2, scalarValuedGaussianKernel3)
    val augmentedGP = GaussianProcess(zeroMean, augmentedCov)


    val relativeTolerance = 0.01
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      asm.statisticalModel.mean.pointSet,
      augmentedGP,
      relativeTolerance,
      interpolator
    )

    val m = lowRankGP.sampleAtPoints(asm.statisticalModel.mean.pointSet).interpolate(NearestNeighborInterpolator())
    val ssmModel = asm.statisticalModel.mean.transform((p : Point[_3D]) => p + m(p))
    val modelView = ui.show(modelGroup, ssmModel, "shapeModel")
    println(ssmModel)
    MeshIO.writeMesh(ssmModel,new File("data/SSMProject2/augmentedASM.stl"))



  }
  }
