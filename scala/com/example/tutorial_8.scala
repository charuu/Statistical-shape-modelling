package com.example

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.{MeshIO, StatisticalModelIO}
import scalismo.mesh.TriangleMesh
import scalismo.statisticalmodel.{DiscreteGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_8 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/bfm.h5")).get
    val gp = model.gp

    val modelGroup = ui.createGroup("modelGroup")
    val ssmView = ui.show(modelGroup, model, "model")
    val sampleDF : DiscreteField[_3D,UnstructuredPointsDomain[_3D], EuclideanVector[_3D]]
    = model.gp.sample

    val sampleGroup = ui.createGroup("sample")
    ui.show(sampleGroup, sampleDF, "discreteSample")
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val contGP = model.gp.interpolate(interpolator)
    val contSample: Field[_3D, EuclideanVector[_3D]] = contGP.sample
    val fullSample = contGP.sampleAtPoints(model.referenceMesh.pointSet)
    val fullSampleView = ui.show(sampleGroup, fullSample, "fullSample")
 //   fullSampleView.remove()
    val singlePointDomain : DiscreteDomain[_3D] =
      UnstructuredPointsDomain(IndexedSeq(model.referenceMesh.pointSet.point(PointId(8156))))

    val singlePointSample = contGP.sampleAtPoints(singlePointDomain)
    ui.show(sampleGroup, singlePointSample, "singlePointSample")

    val referencePointSet = model.referenceMesh.pointSet
    val rightEyePt: Point[_3D] = referencePointSet.point(PointId(4281))
    val leftEyePt: Point[_3D] = referencePointSet.point(PointId(11937))
    val dom = UnstructuredPointsDomain(IndexedSeq(rightEyePt,leftEyePt))
    val marginal : DiscreteGaussianProcess[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = contGP.marginal(dom)
    val sample : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = marginal.sample
    ui.show(sampleGroup, sample, "marginal_sample")
    val noseTipModel : StatisticalMeshModel = model.marginal(IndexedSeq(PointId(8156)))
    val tipSample : TriangleMesh[_3D] = noseTipModel.sample
    // tipSample: TriangleMesh[_3D] = TriangleMesh3D(
    //   scalismo.common.UnstructuredPointsDomain3D@6e15e143,
    //   TriangleList(Vector())
    // )
    println("nb mesh points " + tipSample.pointSet.numberOfPoints)
    // nb mesh points 1
    val middleNose = referencePointSet.point(PointId(9000))
    val nosePtIDs : Iterator[PointId] = referencePointSet.pointsWithId
      .filter( ptAndId => {  // yields tuples with point and ids
        val (pt, id) = ptAndId
        (pt - middleNose).norm > 10
      })
      .map(ptAndId => ptAndId._2) // extract the id's
    val noseModel = model.marginal(nosePtIDs.toIndexedSeq)
    val noseGroup = ui.createGroup("noseModel")
    ui.show(noseGroup, noseModel, "noseModel")
    val defSample = noseModel.gp.sample
    noseModel.gp.pdf(defSample)
    val defSample1 = noseModel.gp.sample
    // defSample1: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = <function1>
    val defSample2 = noseModel.gp.sample
    // defSample2: DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = <function1>

    val logPDF1 = noseModel.gp.logpdf(defSample1)
    // logPDF1: Double = -11.265529462996712
    val logPDF2 = noseModel.gp.logpdf(defSample2)
    // logPDF2: Double = -17.33330521109113

    val moreOrLess = if (logPDF1 > logPDF2) "more" else "less"
    // moreOrLess: String = "more"
    println(s"defSample1 is $moreOrLess likely than defSample2")
  }
}
