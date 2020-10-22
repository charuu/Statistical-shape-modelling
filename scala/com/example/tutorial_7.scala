package com.example

import scalismo.common.{Field, NearestNeighborInterpolator, RealSpace}
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.MeshIO
import scalismo.kernels.{DiagonalKernel, GaussianKernel, PDKernel}
import scalismo.statisticalmodel.GaussianProcess
import scalismo.ui.api.ScalismoUI

object tutorial_7 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val referenceMesh = MeshIO.readMesh(new java.io.File("data/lowResPaola.stl")).get

    val modelGroup = ui.createGroup("gp-model")
    val referenceView = ui.show(modelGroup, referenceMesh, "reference")
    val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))

    val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 10.0)

    val gp = GaussianProcess(zeroMean, DiagonalKernel(scalarValuedGaussianKernel, 3))
    val sampleGroup = ui.createGroup("samples")
    val sample = gp.sampleAtPoints(referenceMesh.pointSet)
    ui.show(sampleGroup, sample, "gaussianKernelGP_sample")
    val interpolatedSample = sample.interpolate(NearestNeighborInterpolator())
    val deformedMesh = referenceMesh.transform((p : Point[_3D]) => p + interpolatedSample(p))
    ui.show(sampleGroup, deformedMesh, "deformed mesh")

  }
}
