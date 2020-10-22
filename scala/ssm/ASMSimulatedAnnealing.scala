package ssm

import java.io.File

import breeze.linalg.DenseVector
import scalismo.geometry.EuclideanVector
import scalismo.io.{ActiveShapeModelIO, ImageIO, MeshIO}
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.statisticalmodel.asm.{FittingConfiguration, ModelTransformations, NormalDirectionSearchPointSampler}
import scalismo.ui.api.ScalismoUI

object ASMSimulatedAnnealing {
  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()
    val modelGroup = ui.createGroup("modelGroup")
    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get
    val model = MeshIO.readMesh(new java.io.File("data/SSMProject2/2.stl")).get

   // val modelGroup = ui.createGroup("pcaModel")

    val ssmModel = asm.statisticalModel.copy(model)
    val modelView = ui.show(ssmModel, "shapeModel")
    val refGroup = ui.createGroup("target")
    val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/9.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")
    val imageView = ui.show(targetGroup, image, "image")

    val searchSampler = NormalDirectionSearchPointSampler(numberOfPoints = 500, searchDistance = 5)
    val config = FittingConfiguration(featureDistanceThreshold = 10, pointDistanceThreshold = 7, modelCoefficientBounds = 5)

    val modelBoundingBox = ssmModel.referenceMesh.boundingBox
    val rotationCenter = modelBoundingBox.origin + modelBoundingBox.extent * 0.5

    // we start with the identity transform
    val translationTransformation = TranslationTransform(EuclideanVector(0, 0, 0))
    val rotationTransformation = RotationTransform(0, 0, 0, rotationCenter)
    val initialRigidTransformation = RigidTransformation(translationTransformation, rotationTransformation)
    val initialModelCoefficients = DenseVector.zeros[Double](ssmModel.rank)
    val initialTransformation = ModelTransformations(initialModelCoefficients, initialRigidTransformation)

    val numberOfIterations = 100
    val i=0
    val asmIterator = asm.fitIterator(image, searchSampler, numberOfIterations, config, initialTransformation)
    val asmIteratorWithVisualization = asmIterator.map(it => {
      it match {
        case scala.util.Success(iterationResult) => {
          modelView.shapeModelTransformationView.poseTransformationView.transformation = iterationResult.transformations.rigidTransform
          modelView.shapeModelTransformationView.shapeTransformationView.coefficients = iterationResult.transformations.coefficients
          println("iteration ", i)
            }
        case scala.util.Failure(error) => System.out.println(error.getMessage)
      }
      it
    })


    val result = asmIteratorWithVisualization.toIndexedSeq.last
    val finalMesh = result.get.mesh
    ui.show(finalMesh,"final")

    MeshIO.writeSTL(finalMesh,new File("data/SSMProject2/3.stl"))


  }
}
