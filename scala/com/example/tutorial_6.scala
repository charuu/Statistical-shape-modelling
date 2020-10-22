package com.example

import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry._
import scalismo.io.MeshIO
import scalismo.kernels.DiscreteMatrixValuedPDKernel
import scalismo.mesh.TriangleMesh
import scalismo.registration.LandmarkRegistration
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object tutorial_6 {
  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val dsGroup = ui.createGroup("datasets")

    val meshFiles = new java.io.File("data/nonAlignedFaces/").listFiles
    val (meshes, meshViews) = meshFiles.map(meshFile => {
      val mesh = MeshIO.readMesh(meshFile).get
      val meshView = ui.show(dsGroup, mesh, "mesh")
      (mesh, meshView) // return a tuple of the mesh and the associated view
    }) .unzip
    val reference = meshes.head
    val toAlign : IndexedSeq[TriangleMesh[_3D]] = meshes.tail
    val pointIds = IndexedSeq(2214, 6341, 10008, 14129, 8156, 47775)
    val refLandmarks = pointIds.map{id => Landmark(s"L_$id", reference.pointSet.point(PointId(id))) }
    val paolaLandmarkViews = refLandmarks.map(lm => ui.show(lm, s"L_${lm.id}"))

    val alignedMeshes = toAlign.map { mesh =>
      val landmarks = pointIds.map{id => Landmark("L_"+id, mesh.pointSet.point(PointId(id)))}
      val LandmarkViews = landmarks.map(lm => ui.show(lm, "L_${lm.id}"))
      val rigidTrans = LandmarkRegistration.rigid3DLandmarkRegistration(landmarks, refLandmarks, center = Point(0,0,0))
      mesh.transform(rigidTrans)
    }

    val defFields = alignedMeshes.map{ m =>
      val deformationVectors = reference.pointSet.pointIds.map{ id : PointId =>
        m.pointSet.point(id) - reference.pointSet.point(id)
      }.toIndexedSeq
//48420 9013
      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](reference.pointSet, deformationVectors)
    }
    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]()
    val continuousFields = defFields.map(f => f.interpolate(interpolator) )
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, continuousFields)


    val res5: EuclideanVector[_3D] = EuclideanVector3D(
      56.99011770072786,0.5270593377396238,89.03113559208577
     )


   print("Point" + res5.toPoint)

    val sampleCov : DiscreteMatrixValuedPDKernel[_3D]
    = gp.cov
    val k = sampleCov.k(PointId(3223),PointId(9013))
    print("Nearby point cov" + k)
    print("gp cov" +sampleCov)
    val model = StatisticalMeshModel(reference, gp.interpolate(interpolator))
    val modelGroup = ui.createGroup("model")
    val ssmView = ui.show(modelGroup, model, "model")









   /* val dc = DataCollection.fromMeshSequence(reference, alignedMeshes)._1.get
    val item0 :DataItem[_3D] = dc.dataItems(0)
    val transform : Transformation[_3D] = item0.transformation
    val modelNonAligned = StatisticalMeshModel.createUsingPCA(dc).get

    val modelGroup2 = ui.createGroup("modelGroup2")
    ui.show(modelGroup2, modelNonAligned, "nonAligned")
    val sample = modelNonAligned.sample();
    ui.show(sample,"sample")*/
  }
}
