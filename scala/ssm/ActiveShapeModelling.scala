package ssm

import breeze.linalg.DenseVector
import scalismo.geometry.{EuclideanVector, IntVector, _3D}
import scalismo.image.DiscreteImageDomain
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.statisticalmodel.asm._
import scalismo.ui.api.ScalismoUI

object ActiveShapeModelling {
  def likelihoodForMesh(asm : ActiveShapeModel, mesh : TriangleMesh[_3D], preprocessedImage: PreprocessedImage) : Double = {

    val ids = asm.profiles.ids

    val likelihoods = for (id <- ids) yield {
      val profile = asm.profiles(id)
      val profilePointOnMesh = mesh.pointSet.point(profile.pointId)
      val featureAtPoint = asm.featureExtractor(preprocessedImage, profilePointOnMesh, mesh, profile.pointId).get
      profile.distribution.logpdf(featureAtPoint)
    }
    likelihoods.sum
  }


  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
    val modelView = ui.show(modelGroup, asm.statisticalModel, "shapeModel")

    val profiles = asm.profiles
    profiles.map(profile => {
      val pointId = profile.pointId
      val distribution = profile.distribution
    })

    val i = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/1.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")
    val imageView = ui.show(targetGroup, i, "image")
    val preprocessedImage = asm.preprocessor(i)

    val gpDomain = i.domain.boundingBox
    val gpApproximationGrid = DiscreteImageDomain(gpDomain, size = IntVector(32, 32, 64))

    val image = i.resample(gpApproximationGrid,3,0.00F)

    val point1 = gpApproximationGrid.origin + EuclideanVector(10.0, 10.0, 10.0)
    val profile = asm.profiles.head
    val feature1 : DenseVector[Double] = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
    val point2 = gpApproximationGrid.origin + EuclideanVector(20.0, 10.0, 10.0)


    val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 100, searchDistance = 3)
    val config = FittingConfiguration(featureDistanceThreshold = 3, pointDistanceThreshold = 5, modelCoefficientBounds = 3)
    // make sure we rotate around a reasonable center point
    val modelBoundingBox = asm.statisticalModel.referenceMesh.boundingBox
    val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

    // we start with the identity transform
    val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))
    val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
    val initialRigidTransformation = RigidTransformation(translationTransformation, rotationTransformation)
    val initialModelCoefficients = DenseVector.zeros[Double](asm.statisticalModel.rank)
    val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

    val numberOfIterations = 20

    val asmIterator = asm.fitIterator(image, searchSampler, numberOfIterations, config, initialTransformation)
    val asmIteratorWithVisualization = asmIterator.map(it => {
      it match {
        case scala.util.Success(iterationResult) => {
          modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
        }
        case scala.util.Failure(error) => System.out.println(error.getMessage)
      }
      it
    })
    val result = asmIteratorWithVisualization.toIndexedSeq.last
    val finalMesh = result.get.mesh
    ui.show(finalMesh,"final")
    val featureVec1 = asm.featureExtractor(preprocessedImage, point1, asm.statisticalModel.mean, profile.pointId).get
    val featureVec2 = asm.featureExtractor(preprocessedImage, point2, asm.statisticalModel.mean, profile.pointId).get


    val probabilityPoint1 = profile.distribution.logpdf(featureVec1)
    val probabilityPoint2 = profile.distribution.logpdf(featureVec2)

    val sampleMesh1 = asm.statisticalModel.sample
    val sampleMesh2 = asm.statisticalModel.sample
    ui.show(sampleMesh1,"mesh1")
    ui.show(sampleMesh2,"mesh2")
    println("Likelihood for mesh 1 = " + likelihoodForMesh(asm, finalMesh, preprocessedImage))
    println("Likelihood for mesh 2 = " + likelihoodForMesh(asm, sampleMesh2, preprocessedImage))


  }

}
