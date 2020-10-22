package com.example

import scalismo.common._
import scalismo.geometry._
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.kernels.{DiagonalKernel, GaussianKernel, MatrixValuedPDKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object FemurKernel3 {
    case class ChangePointKernel(kernel1 : MatrixValuedPDKernel[_3D], kernel2 : MatrixValuedPDKernel[_3D])
      extends MatrixValuedPDKernel[_3D]() {

        override def domain = RealSpace[_3D]
        val outputDim = 3
        val pt :Point[_3D]= Point3D(8.251136779785156,22.13014030456543,-10.564556121826172)
        def s(p: Point[_3D]) =  1.0 / (1.0 + math.exp(-pt(0)))
        def k(x: Point[_3D], y: Point[_3D]) = {
            val sx = s(x)
            val sy = s(y)
            kernel1(x,y) * sx * sy + kernel2(x,y) * (1-sx) * (1-sy)
        }

    }
    def main(args: Array[String]): Unit = {


        implicit val rng = Random(42)
        val ui = ScalismoUI()

        //mesh
        val pcaModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/femur-asm.h5")).get
        val gpSSM = pcaModel.gp.interpolate(NearestNeighborInterpolator())

        val gk1 = DiagonalKernel(GaussianKernel[_3D](100.0)*40.00, 3)
        val gk2 = DiagonalKernel(GaussianKernel[_3D](40.0)*2.00, 3)
        val changePointKernel = ChangePointKernel(gk1, gk2)
        val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
        val gpCP = GaussianProcess(zeroMean, changePointKernel)


        val relativeTolerance = 0.01
        val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

        val lowRankAugmentedGP = LowRankGaussianProcess.approximateGPCholesky(
            pcaModel.referenceMesh.pointSet,
            gpCP,
            relativeTolerance,
            interpolator
        )

        val augmentedSSM = StatisticalMeshModel(pcaModel.referenceMesh, lowRankAugmentedGP)
        ui.show(augmentedSSM ,"changepointssm")
        ui.show(lowRankAugmentedGP.sampleAtPoints(pcaModel.referenceMesh.pointSet), "Gaussian kernel")
        val interpolatedSample =  lowRankAugmentedGP.sampleAtPoints(pcaModel.referenceMesh.pointSet).interpolate(NearestNeighborInterpolator())
        val deformedMesh = pcaModel.referenceMesh.transform((p : Point[_3D]) => p + interpolatedSample(p))
        ui.show(deformedMesh, "deformed mesh")


    }
}