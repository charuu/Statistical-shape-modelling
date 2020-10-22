package com.example

import scalismo.geometry.{Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO, StatisticalModelIO}
import scalismo.ui.api.ScalismoUI

object ShapeCompletion {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val targetMesh = MeshIO.readMesh(new java.io.File("data/paritals_step3/VSD.Right_femur.XX.XX.OT.101148/VSD.Right_femur.XX.XX.OT.101148.0.stl")).get

    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File("data/femur-asm.h5")).get

    val modelGroup = ui.createGroup("model")
    val targetGroup = ui.createGroup("model")
    ui.show(targetGroup,targetMesh,"target")
    ui.show(modelGroup,model,"model")


    val referenceLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/modelLandmarks.json")).get
    val referencePoints : Seq[Point[_3D]] = referenceLandmarks.map(lm => lm.point)
    val referenceLandmarkViews = referenceLandmarks.map(lm => ui.show(modelGroup, lm, s"lm-${lm.id}"))


    val noselessLandmarks = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/noselessLandmarks.json")).get
    val noselessPoints : Seq[Point[_3D]] = noselessLandmarks.map(lm => lm.point)
    val noselessLandmarkViews = noselessLandmarks.map(lm => ui.show(targetGroup, lm, s"lm-${lm.id}"))
  }
  }
