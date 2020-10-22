package com.example

import scalismo.geometry._3D
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.ui.api.ScalismoUI

object FemurModelConstruction {


  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/alignedFemur/").listFiles.sortBy(f => f.getName())
    val meshLmFiles = new java.io.File("data/TransformedLMs/").listFiles.sortBy(f => f.getName())

    val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}
    val mesheslandmarks= meshLmFiles.map{lm => LandmarkIO.readLandmarksJson[_3D](lm).get}

   val reference = meshes(0)
   val reflandmark = mesheslandmarks(0)

    (0 until mesheslandmarks.length).map{i => ui.show(mesheslandmarks(i), "transformed landmark points")}


    ui.show(reference,"reference")



  }




}
