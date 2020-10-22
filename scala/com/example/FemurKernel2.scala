package com.example

import scalismo.common._
import scalismo.geometry._
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel, PDKernel}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI
import scalismo.utils.Random

object FemurKernel2 {
    def main(args: Array[String]): Unit = {


        implicit val rng = Random(42)
        val ui = ScalismoUI()

        //mesh
        val mesh  = MeshIO.readMesh(new java.io.File("data/femurModel1.stl")).get

        ui.show(mesh ,"mesh")


        val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
        val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
        val scalarValuedGaussianKernel1 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
        val scalarValuedGaussianKernel2 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
        val gp = GaussianProcess(zeroMean,  DiagonalKernel(scalarValuedGaussianKernel,scalarValuedGaussianKernel1,scalarValuedGaussianKernel2))

        val relativeTolerance = 0.01
        val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

        val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
            mesh.pointSet,
            gp,
            relativeTolerance,
            interpolator
        )


        ui.show(lowRankGP.sampleAtPoints(mesh.pointSet), "gaussianKernelGP_sample")
        ui.show(StatisticalMeshModel(mesh, lowRankGP), "group")
        val interpolatedSample =  lowRankGP.sampleAtPoints(mesh.pointSet).interpolate(NearestNeighborInterpolator())
        val deformedMesh = mesh.transform((p : Point[_3D]) => p + interpolatedSample(p))
        ui.show(deformedMesh, "deformed mesh")

    }
}