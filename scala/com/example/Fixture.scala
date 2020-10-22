package com.example

import scalismo.geometry.Point
import scalismo.io.MeshIO
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.statisticalmodel.dataset.DataCollection
import scalismo.ui.api.ScalismoUI

object Fixture {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val nonAlignedFaces = new java.io.File("data/meshes/").listFiles.sortBy(_.getName).map { f => MeshIO.readMesh(f).get }.toIndexedSeq
    val ref = nonAlignedFaces.head
    val dataset = nonAlignedFaces.tail

    val aligendDataset = dataset.map { d =>
      val trans = LandmarkRegistration.rigid3DLandmarkRegistration((d.pointSet.points zip ref.pointSet.points).toIndexedSeq, Point(0, 0, 0))
      d.transform(trans)
    }

    val trainingSet = aligendDataset.drop(3)
    val testingSet = aligendDataset.take(3)

    val dc = DataCollection.fromMeshSequence(ref, trainingSet)._1.get
    val pcaModel = StatisticalMeshModel.createUsingPCA(dc).get

    val testDC = DataCollection.fromMeshSequence(pcaModel.referenceMesh, testingSet)._1.get




  }

}
