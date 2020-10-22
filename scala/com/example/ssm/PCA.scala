package com.example.ssm

import scalismo.geometry._3D
import scalismo.io.{ActiveShapeModelIO, ImageIO, LandmarkIO}
import scalismo.ui.api.ScalismoUI

object PCA {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)
    val ui = ScalismoUI()

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get

    val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/37.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")
    ui.show(targetGroup, image, "image")

    val MOlandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/handedData/targets/Landmarks/modelLandmarks.json")).get
    ui.show(MOlandmark, "modelLandmarks")
    val modelLandmarkPoints = MOlandmark.seq.map { l => asm.statisticalModel.mean.pointSet.findClosestPoint(l.point).id }
    print(modelLandmarkPoints)



  }
}
