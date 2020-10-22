import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.io._
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.algorithms.MetropolisHastings
import scalismo.sampling.evaluators.ProductEvaluator
import scalismo.sampling.proposals.MixtureProposal
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.statisticalmodel.asm._
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object IntensityliklihoodASM2 {


  case class Parameters(translationParameters: EuclideanVector[_3D],
                        rotationParameters: (Double, Double, Double),
                        modelCoefficients: DenseVector[Double],profiles :Profiles)
  case class Sample(generatedBy : String, parameters : Parameters, rotationCenter: Point[_3D]) {
    def poseTransformation : RigidTransformation[_3D] = {

      val translation = TranslationTransform(parameters.translationParameters)
      val rotation = RotationTransform(
        parameters.rotationParameters._1,
        parameters.rotationParameters._2,
        parameters.rotationParameters._3,
        rotationCenter
      )
      RigidTransformation(translation, rotation)
    }
  }

  case class PriorEvaluator(model: StatisticalMeshModel)
    extends DistributionEvaluator[Sample] {
    val intensityPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
    val translationPrior = breeze.stats.distributions.Gaussian(0.0, 5.0)
    val rotationPrior = breeze.stats.distributions.Gaussian(0, 0.1)

    override def logValue(sample: Sample): Double = {
      model.gp.logpdf(sample.parameters.modelCoefficients) +
        translationPrior.logPdf(sample.parameters.translationParameters.x) +
        translationPrior.logPdf(sample.parameters.translationParameters.y) +
        translationPrior.logPdf(sample.parameters.translationParameters.z) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._1) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._2) +
        rotationPrior.logPdf(sample.parameters.rotationParameters._3)
    }
  }

  def computeCenterOfMass(mesh : TriangleMesh[_3D]) : Point[_3D] = {
    val normFactor = 1.0 / mesh.pointSet.numberOfPoints
    mesh.pointSet.points.foldLeft(Point(0, 0, 0))((sum, point) => sum + point.toVector * normFactor)
  }

  def marginalizeModelForCorrespondences(model: StatisticalMeshModel,
                                         correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
  : (StatisticalMeshModel, Seq[(PointId, Point[_3D], MultivariateNormalDistribution)]) = {

    val (modelIds, _, _) = correspondences.unzip3
    val marginalizedModel = model.marginal(modelIds.toIndexedSeq)
    val newCorrespondences = correspondences.map(idWithTargetPoint => {
      val (id, targetPoint, uncertainty) = idWithTargetPoint
      val modelPoint = model.referenceMesh.pointSet.point(id)
      val newId = marginalizedModel.referenceMesh.pointSet.findClosestPoint(modelPoint).id
      (newId, targetPoint, uncertainty)
    })
    (marginalizedModel, newCorrespondences)
  }


  case class CorrespondenceEvaluator2(asm:ActiveShapeModel,model: StatisticalMeshModel,
                                     correspondences : Seq[(PointId, Point[_3D], MultivariateNormalDistribution)],image : scalismo.statisticalmodel.asm.PreprocessedImage )
    extends DistributionEvaluator[Sample] {

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    def logValue(sample: Sample): Double = {

      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val ids = sample.parameters.profiles.ids

      val likelihoods = for (id <- ids) yield {
        val profile = sample.parameters.profiles(id)
        val profilePointOnMesh = currModelInstance.pointSet.point(profile.pointId)
        val featureAtPoint = asm.featureExtractor(image, profilePointOnMesh, model.mean, profile.pointId).get
        profile.distribution.logpdf(featureAtPoint)
      }
      likelihoods.sum
    }
  }

  case class TranslationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample]  with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution( DenseVector.zeros(3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample= {
      val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(perturbationDistr.sample())
      val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
      sample.copy(generatedBy = s"TranlationUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.translationParameters - from.parameters.translationParameters
      perturbationDistr.logpdf(residual.toBreezeVector)
    }
  }

  case class RotationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample]  with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * stddev * stddev)
    def propose(sample: Sample): Sample= {

      val perturbation = perturbationDistr.sample()
      val newRotationParameters = (
        sample.parameters.rotationParameters._1 + perturbation(0),
        sample.parameters.rotationParameters._2 + perturbation(1),
        sample.parameters.rotationParameters._3 + perturbation(2)
      )
      val newParameters = sample.parameters.copy(rotationParameters = newRotationParameters)
      sample.copy(generatedBy = s"RotationUpdateProposal ($stddev)", parameters = newParameters)
    }
    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = DenseVector(
        to.parameters.rotationParameters._1 - from.parameters.rotationParameters._1,
        to.parameters.rotationParameters._2 - from.parameters.rotationParameters._2,
        to.parameters.rotationParameters._3 - from.parameters.rotationParameters._3
      )
      perturbationDistr.logpdf(residual)
    }
  }

  case class ShapeUpdateProposal(paramVectorSize : Int, stddev: Double)
    extends ProposalGenerator[Sample]  with TransitionProbability[Sample] {

    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros(paramVectorSize),
      DenseMatrix.eye[Double](paramVectorSize) * stddev * stddev
    )


    override def propose(sample: Sample): Sample = {

      implicit val rng = scalismo.utils.Random(42)
      val perturbation = perturbationDistr.sample()
      val newParameters = sample.parameters.copy(modelCoefficients = sample.parameters.modelCoefficients + perturbationDistr.sample)
      sample.copy(generatedBy = s"ShapeUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.modelCoefficients - from.parameters.modelCoefficients
      perturbationDistr.logpdf(residual)
    }
  }


  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)

    val ui = ScalismoUI()

    val asm = ActiveShapeModelIO.readActiveShapeModel(new java.io.File("data/handedData/femur-asm.h5")).get
    val modelGroup = ui.createGroup("modelGroup")
  //  val model =  MeshIO.readMesh(new java.io.File("data/SSMProject2/3.stl")).get
    val mesh = MeshIO.readMesh(new java.io.File("data/SSMProject2/2.stl")).get
    val ssmModel = asm.statisticalModel.copy(mesh)
    println(ssmModel.rank)
    val modelView = ui.show(modelGroup, ssmModel, "shapeModel")

    val image = ImageIO.read3DScalarImage[Short](new java.io.File("data/handedData/targets/1.nii")).get.map(_.toFloat)
    val targetGroup = ui.createGroup("target")
    val imageView = ui.show(targetGroup, image, "image")
    val preprocessedImage = asm.preprocessor(image)
    val gpDomain = image.domain.boundingBox

    val initialParameters = Parameters(
      EuclideanVector(0, 0, 0),
      (0.0, 0.0, 0.0),
      DenseVector.zeros[Double](ssmModel.rank),
      asm.profiles
    )
      val landmarkNoiseVariance = 12.0
    val uncertainty = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * landmarkNoiseVariance
    )

    val correspondences = ssmModel.referenceMesh.pointSet.pointsWithId.zip(image.domain.points).map(modelIdWithTargetPoint => {
      val modelId = modelIdWithTargetPoint._1._2
      val targetPoint =  modelIdWithTargetPoint._2
      (modelId, targetPoint, uncertainty)
    }).toSeq

    val initialSample = Sample("initial", initialParameters, computeCenterOfMass(ssmModel.referenceMesh))
    val likelihoodEvaluator = CorrespondenceEvaluator2(asm,ssmModel,correspondences, preprocessedImage)
    val priorEvaluator = PriorEvaluator(ssmModel)
    val posteriorEvaluator = ProductEvaluator(priorEvaluator,likelihoodEvaluator)

    val rotationUpdateProposal = RotationUpdateProposal(0.01)
    val translationUpdateProposal = TranslationUpdateProposal(1.0)
    val shapeUpdateProposal = ShapeUpdateProposal(ssmModel.rank, 0.1)
    println(ssmModel.rank)

    val generator = MixtureProposal.fromProposalsWithTransition(
      (0.3, rotationUpdateProposal),
      (0.3, translationUpdateProposal),
      (0.4,shapeUpdateProposal))
    val chain = MetropolisHastings(generator, posteriorEvaluator)
   // val logger = new Logger()
    val mhIterator = chain.iterator(initialSample)

    val samplingIterator = for((sample, iteration) <- mhIterator.zipWithIndex) yield {
      println("iteration " + iteration)
      if (iteration % 500 == 0) {
        modelView.shapeModelTransformationView.poseTransformationView.transformation = sample.poseTransformation
        modelView.shapeModelTransformationView.shapeTransformationView.coefficients = sample.parameters.modelCoefficients

      }
      sample
    }

    val samples = samplingIterator.drop(1000).take(3000).toIndexedSeq

 //   println(logger.acceptanceRatios())

    val bestSample = samples.maxBy(posteriorEvaluator.logValue)
    val bestFit = ssmModel.instance(bestSample.parameters.modelCoefficients).transform(bestSample.poseTransformation)
    val resultGroup = ui.createGroup("result")
    ui.show(resultGroup, bestFit, "best fit")

    MeshIO.writeSTL(bestFit,new File("data/SSMProject2/3.stl"))





  }
}
