package com.example

import java.io.File

import scalismo.geometry.{Point, _3D}
import scalismo.io.{LandmarkIO, MeshIO}
import scalismo.registration.LandmarkRegistration
import scalismo.ui.api.ScalismoUI

object femurRigidAligning {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/SSMProject1/meshes/").listFiles.sortBy(f => f.getName())
    val meshLmFiles = new java.io.File("data/SSMProject1/landmarks/").listFiles.sortBy(f => f.getName())

    val reference = MeshIO.readMesh(new java.io.File("data/femur.stl")).get
    val reflandmark = LandmarkIO.readLandmarksJson[_3D](new java.io.File("data/femur.json")).get

    val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}
    val mesheslandmarks= meshLmFiles.map{f => LandmarkIO.readLandmarksJson[_3D](f).get}

    val bestTransform = (0 until meshes.length ).map{i => LandmarkRegistration.rigid3DLandmarkRegistration(mesheslandmarks(i), reflandmark, center = Point(0, 0, 0))}
    val alignedMeshes = (0 until meshes.length).map{i => meshes(i).transform(bestTransform(i))}

    val transLM  =(0 until mesheslandmarks.length).map{i => mesheslandmarks(i).map{lm => lm.copy(point = bestTransform(i)(lm.point))}}

    (0 until mesheslandmarks.length).map{i => ui.show(transLM(i), "transformed landmark points")}
    ui.show(reference, "Reference meshes")
    // (0 until mesheslandmarks.length).map{i => ui.show(mesheslandmarks(i),"landmarks"+i)}
    //  (0 until meshes.length).map{i =>ui.show(meshes(i), "sample meshes")}
    (0 until meshes.length).map{i =>ui.show(alignedMeshes(i), "Aligned meshes")}

    (0 until mesheslandmarks.length).map{i => LandmarkIO.writeLandmarksJson(transLM(i), new File("data/SSMProject1/TransformedLms/transformedLM" +i+".json"))}

  //  MeshIO.writeVTK(reference, new File("data/reference.vtk"))
    (0 until meshes.length).foreach { i: Int => MeshIO.writeVTK(alignedMeshes(i), new File("data/SSMProject1/alignedFemur/femur_" + i + ".vtk")) }

  //  val meshVtkFiles = new java.io.File("data/alignedFemur/").listFiles.sortBy(f => f.getName())
  //  val refVtk = new java.io.File("data/alignedFemur/reference.vtk")

 //   val VtkFiles: IndexedSeq[TriangleMesh[_3D]] = meshVtkFiles.map{f => MeshIO.readMesh(f).get}


 //   val ref = MeshIO.readMesh(refVtk).get

  /*  val defFields = alignedMeshes.map{ m =>
      val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
        if ((m.pointSet.points.length - 1 >= id.id) && (reference.pointSet.points.length - 1 >= id.id)) {
          m.pointSet.point(id) - reference.pointSet.point(id)
        }
        else EuclideanVector3D(0,0,0)
      }.toIndexedSeq

      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator) )

   val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)

   // val pcaModel = StatisticalMeshModel(reference, gp.interpolate(interpolator))
    val modelGroup = ui.createGroup("model")
  //  val ssmView = ui.show(pcaModel, "model") */

  /*  val zeroMean = Field(RealSpace[_3D], (pt:Point[_3D]) => EuclideanVector(0,0,0))
    val scalarValuedGaussianKernel : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
    val scalarValuedGaussianKernel1 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
    val scalarValuedGaussianKernel2 : PDKernel[_3D]= GaussianKernel(sigma = 100.0) * 10.0
    val gp2 = GaussianProcess(zeroMean,  DiagonalKernel(scalarValuedGaussianKernel,scalarValuedGaussianKernel1,scalarValuedGaussianKernel2))


    val relativeTolerance = 0.01
    val interpolator2 = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()

    val lowRankGP = LowRankGaussianProcess.approximateGPCholesky(
      reference.pointSet,
      gp2,
      relativeTolerance,
      interpolator2
    )
    ui.show(lowRankGP.sampleAtPoints(reference.pointSet), "gaussianKernelGP_sample")
   val GPSSM =StatisticalMeshModel(reference, lowRankGP)
    ui.show(GPSSM, "group")
    val interpolatedSample =  lowRankGP.sampleAtPoints(reference.pointSet).interpolate(NearestNeighborInterpolator())
    val GP = ui.createGroup("GP")
    val deformedMesh = reference.transform((p : Point[_3D]) => p + interpolatedSample(p))
    ui.show(GP,deformedMesh, "deformed mesh")
*/

    //  val modelGroup = ui.createGroup("model")
  //  val ssmView = ui.show(modelGroup, pcaModel, "model")
  //  StatismoIO.writeStatismoMeshModel(GPSSM,new File("data/model6.h5"),"/",StatismoIO.StatismoVersion.v090)
  ///  (0 until meshes.length).map{i =>StatismoIO.writeStatismoMeshModel(alignedMeshes(i), new File("data/aligned_meshes/femur_4.h5"))}
    // val meshes = meshFiles.map{f => MeshIO.readMesh(f).get}
    //   val mesheslandmarks= meshLmFiles.map{f => LandmarkIO.readLandmarksJson[_3D](f).get}
    //   val bestTransform = (0 until meshes.length ).map{i => LandmarkRegistration.rigid3DLandmarkRegistration(mesheslandmarks(i), reflandmark, center = Point(0, 0, 0))}
    //  val alignedMeshes = (0 until meshes.length).map{i => meshes(i).transform(bestTransform(i))}
    // (0 until mesheslandmarks.length).map{i => ui.show(mesheslandmarks(i),"landmarks"+i)}
    //   (0 until alignedMeshes.length).map{i => ui.show(alignedMeshes(i),"landmarks"+i)}
    //  println("Start writing landmarks at `data/aligned/landmarks/`")
    // (0 until 50).foreach { i: Int => LandmarkIO.writeLandmarksJson(mesheslandmarks(i), new File("data/aligned_meshes_" +i+".json")) }

  }
}