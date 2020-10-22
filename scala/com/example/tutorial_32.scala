package com.example

import scalismo.common._
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io.StatisticalModelIO
import scalismo.mesh.TriangleMesh
import scalismo.registration.Transformation
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_32 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val faceModel = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/bfm.h5")).get
    val modelGroup = ui.createGroup("model")
    val sampleGroup = ui.createGroup("samples")

    val meanFace : TriangleMesh[_3D] = faceModel.mean
    ui.show(sampleGroup, meanFace, "meanFace")

    val sampledFace : TriangleMesh[_3D] = faceModel.sample
    ui.show(sampleGroup, sampledFace, "randomFace")

    val reference : TriangleMesh[_3D] = faceModel.referenceMesh
    val faceGP : DiscreteLowRankGaussianProcess[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceModel.gp

    val meanDeformation : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceGP.mean
    val sampleDeformation : DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]] = faceGP.sample
   // ui.show(sampleGroup, meanDeformation, "meanField")
    ui.show(modelGroup, reference, "referenceFace")
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

   // ui.show(sampleGroup, sampleDeformation, "sampleField")
  //  val continuousMeanDeformationField = sampleDeformation.interpolate(interpolator)
    //gp
    val continuousgp = faceGP.interpolate(interpolator)

//    val sampleTransformation = Transformation((pt : Point[_3D]) => pt + continuousMeanDeformationField(pt))
//    val sampleMesh = reference.transform(sampleTransformation)
//    ui.show(modelGroup, sampleMesh, "sampleMesh")

 //   val contSample: Field[_3D, EuclideanVector[_3D]] = continuousgpDeformationField.sample
    val fullSample = continuousgp.sampleAtPoints(reference.pointSet)
    val sa= fullSample.interpolate(interpolator)
    val sampleTransformation = Transformation((pt : Point[_3D]) => pt + sa(pt))
    val sampleMesh = reference.transform(sampleTransformation)
    val fullSampleView = ui.show(sampleGroup, sampleMesh, "gpSample")
    val singlePointDomain : DiscreteDomain[_3D] =
      UnstructuredPointsDomain(IndexedSeq(sampleMesh.pointSet.point(PointId(8156))))
    val singlePointSample = continuousgp.sampleAtPoints(singlePointDomain)
   // ui.show(sampleGroup, singlePointSample, "singlePointSample")

    val middleNose = sampleMesh.pointSet.point(PointId(8152))
    val nosePtIDs : Iterator[PointId] = sampleMesh.pointSet.pointsWithId
      .filter( ptAndId => {  // yields tuples with point and ids
        val (pt, id) = ptAndId
        (middleNose - pt).norm < 40
      })
      .map(ptAndId => ptAndId._2) // extract the id's

    val noseTipModel : StatisticalMeshModel = faceModel.marginal(nosePtIDs.toIndexedSeq)
    val tipSample : TriangleMesh[_3D] = noseTipModel.sample
    ui.show(tipSample, "noseModel")



    val fullSample2 = continuousgp.sampleAtPoints(reference.pointSet)
    val sa2= fullSample2.interpolate(interpolator)
    val sampleTransformation2 = Transformation((pt : Point[_3D]) => pt + sa2(pt))
    val sampleMesh2 = reference.transform(sampleTransformation2)
    ui.show(sampleGroup, sampleMesh2, "gpSample2")
    val singlePointDomain2 : DiscreteDomain[_3D] =
      UnstructuredPointsDomain(IndexedSeq(sampleMesh2.pointSet.point(PointId(8156))))
    val singlePointSample2 = continuousgp.sampleAtPoints(singlePointDomain2)
    val middleNose2 = sampleMesh2.pointSet.point(PointId(8152))
    val nosePtIDs2 : Iterator[PointId] = sampleMesh2.pointSet.pointsWithId
      .filter( ptAndId => {  // yields tuples with point and ids
        val (pt, id) = ptAndId
        (middleNose2 - pt).norm < 40
      })
      .map(ptAndId => ptAndId._2) // extract the id's

    val noseTipModel2 : StatisticalMeshModel = faceModel.marginal(nosePtIDs2.toIndexedSeq)
    val tipSample2 : TriangleMesh[_3D] = noseTipModel2.sample
    ui.show(tipSample2, "noseModel")


    val fullSample3 = continuousgp.sampleAtPoints(reference.pointSet)
    val sa3= fullSample3.interpolate(interpolator)
    val sampleTransformation3 = Transformation((pt : Point[_3D]) => pt + sa3(pt))
    val sampleMesh3 = reference.transform(sampleTransformation3)
     ui.show(sampleGroup, sampleMesh3, "gpSample3")
    val singlePointDomain3 : DiscreteDomain[_3D] =
      UnstructuredPointsDomain(IndexedSeq(sampleMesh3.pointSet.point(PointId(8156))))
    val singlePointSample3 = continuousgp.sampleAtPoints(singlePointDomain3)
   // ui.show(sampleGroup, singlePointSample3, "singlePointSample")

    val fullSample4 = continuousgp.sampleAtPoints(reference.pointSet)
    val sa4= fullSample4.interpolate(interpolator)
    val sampleTransformation4 = Transformation((pt : Point[_3D]) => pt + sa4(pt))
    val sampleMesh4 = reference.transform(sampleTransformation4)
     ui.show(sampleGroup, sampleMesh4, "gpSample4")

    val fullSample5 = continuousgp.sampleAtPoints(reference.pointSet)
    val sa5= fullSample5.interpolate(interpolator)
    val sampleTransformation5 = Transformation((pt : Point[_3D]) => pt + sa5(pt))
    val sampleMesh5 = reference.transform(sampleTransformation5)
   ui.show(sampleGroup, sampleMesh5, "gpSample5")


  }
}
