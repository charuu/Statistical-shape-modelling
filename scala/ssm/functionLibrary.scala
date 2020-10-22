/*
package ssm

import breeze.linalg.{DenseMatrix, DenseVector}
import scalismo.common.PointId
import scalismo.geometry.{EuclideanVector, Point, _3D}
import scalismo.mesh.TriangleMesh
import scalismo.registration.{RigidTransformation, RotationTransform, TranslationTransform}
import scalismo.sampling.{DistributionEvaluator, ProposalGenerator, TransitionProbability}
import scalismo.sampling.loggers.AcceptRejectLogger
import scalismo.statisticalmodel.{MultivariateNormalDistribution, StatisticalMeshModel}
import scalismo.statisticalmodel.asm.{ActiveShapeModel, Profiles}
import scalismo.utils.Memoize

class Logger extends AcceptRejectLogger[Sample] {
  private val numAccepted = collection.mutable.Map[String, Int]()
  private val numRejected = collection.mutable.Map[String, Int]()

  override def accept(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
    val numAcceptedSoFar = numAccepted.getOrElseUpdate(sample.generatedBy, 0)
    numAccepted.update(sample.generatedBy, numAcceptedSoFar + 1)

  }

  override def reject(current: Sample,
                      sample: Sample,
                      generator: ProposalGenerator[Sample],
                      evaluator: DistributionEvaluator[Sample]
                     ): Unit = {
    val numRejectedSoFar = numRejected.getOrElseUpdate(sample.generatedBy, 0)
    numRejected.update(sample.generatedBy, numRejectedSoFar + 1)
  }


  def acceptanceRatios(): Map[String, Double] = {
    val generatorNames = numRejected.keys.toSet.union(numAccepted.keys.toSet)
    val acceptanceRatios = for (generatorName <- generatorNames) yield {
      val total = (numAccepted.getOrElse(generatorName, 0)
        + numRejected.getOrElse(generatorName, 0)).toDouble
      (generatorName, numAccepted.getOrElse(generatorName, 0) / total)
    }
    acceptanceRatios.toMap
  }
}



object functionLibrary {
  case class Parameters(translationParameters: EuclideanVector[_3D],
                        rotationParameters: (Double, Double, Double),
                        modelCoefficients: DenseVector[Double], profiles: Profiles)

  case class Sample(generatedBy: String, parameters: Parameters, rotationCenter: Point[_3D]) {
    //val profilesS = profiles
    def shapeTransformation: DenseVector[Double] = {
      val coefficients = parameters.modelCoefficients
      return coefficients
    }



    def poseTransformation: RigidTransformation[_3D] = {

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

    val translationPrior = breeze.stats.distributions.Gaussian(0.0, 2.0)
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


  def computeCenterOfMass(mesh: TriangleMesh[_3D]): Point[_3D] = {
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

  case class CorrespondenceEvaluator(model: StatisticalMeshModel,
                                     correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)])
    extends DistributionEvaluator[Sample] {

    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    override def logValue(sample: Sample): Double = {

      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val likelihoods = correspondences.map(correspondence => {
        val (id, targetPoint, uncertainty) = correspondence
        if (id.id <= currModelInstance.pointSet.points.length) {
          val modelInstancePoint = currModelInstance.pointSet.point(id)
          val observedDeformation = targetPoint - modelInstancePoint
          uncertainty.logpdf(observedDeformation.toBreezeVector)
        } else {
          uncertainty.logpdf(DenseVector.zeros(3))
        }
      })

      val loglikelihood = likelihoods.sum
      loglikelihood
    }


  }


  case class TranslationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample] with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(DenseVector.zeros(3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample = {
      val newTranslationParameters = sample.parameters.translationParameters + EuclideanVector.fromBreezeVector(perturbationDistr.sample())
      val newParameters = sample.parameters.copy(translationParameters = newTranslationParameters)
      sample.copy(generatedBy = "TranslationUpdateProposal ($stddev)", parameters = newParameters)
    }

    override def logTransitionProbability(from: Sample, to: Sample) = {
      val residual = to.parameters.translationParameters - from.parameters.translationParameters
      perturbationDistr.logpdf(residual.toBreezeVector)
    }
  }

  case class RotationUpdateProposal(stddev: Double) extends
    ProposalGenerator[Sample] with TransitionProbability[Sample] {

    implicit val rng = scalismo.utils.Random(42)
    val perturbationDistr = new MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * stddev * stddev)

    def propose(sample: Sample): Sample = {

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

  case class CachedEvaluator[A](evaluator: DistributionEvaluator[A]) extends DistributionEvaluator[A] {
    val memoizedLogValue = Memoize(evaluator.logValue, 10)

    override def logValue(sample: A): Double = {
      memoizedLogValue(sample)
    }
  }

  case class ShapeUpdateProposal(paramVectorSize: Int, stddev: Double)
    extends ProposalGenerator[Sample] with TransitionProbability[Sample] {

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

  case class IntensityEvaluator(asm: ActiveShapeModel, model: StatisticalMeshModel,
                                correspondences: Seq[(PointId, Point[_3D], MultivariateNormalDistribution)], image: scalismo.statisticalmodel.asm.PreprocessedImage)
    extends DistributionEvaluator[Sample] {
    val (marginalizedModel, newCorrespondences) = marginalizeModelForCorrespondences(model, correspondences)

    def logValue(sample: Sample): Double = {
      val currModelInstance = marginalizedModel.instance(sample.parameters.modelCoefficients).transform(sample.poseTransformation)

      val ids = sample.parameters.profiles.ids


      val likelihoods = for (id <- ids) yield {
        val profile = sample.parameters.profiles(id)
        val profilePointOnMesh = model.referenceMesh.pointSet.point(profile.pointId)
        val featureAtPoint = asm.featureExtractor(image, profilePointOnMesh, model.mean, profile.pointId).get
        profile.distribution.logpdf(featureAtPoint)
      }
      val loglikelihood = likelihoods.sum
      loglikelihood
    }
  }
}
*/
