import scalismo.sampling.proposals.MixtureProposal
import ssm.intensityEvaluator.{RandomWalkProposal, Sample}


object MCMC {


  def main(args: Array[String]): Unit = {
    scalismo.initialize()
    implicit val rng = scalismo.utils.Random(42)
    val mu = -5
    val sigma = 17

    val trueDistribution = breeze.stats.distributions.Gaussian(mu, sigma)
    val data = for (_ <- 0 until 100) yield {
      trueDistribution.draw()
    }

    //  val posteriorEvaluator = ProductEvaluator(PriorEvaluator, LikelihoodEvaluator(data))

    val smallStepProposal = RandomWalkProposal(3.0, 1.0)
    val largeStepProposal = RandomWalkProposal(9.0, 3.0)
    val generator = MixtureProposal.fromProposalsWithTransition[Sample](
      (0.8, smallStepProposal),
      (0.2, largeStepProposal)
    )


 //   val chain = MetropolisHastings(generator, posteriorEvaluator)
    /*val initialSample = Sample(Parameters(0.0, 10.0), generatedBy="initial")
    val mhIterator = chain.iterator(initialSample)

    val samples = mhIterator.drop(1000).take(5000).toIndexedSeq
    val estimatedMean = samples.map(sample => sample.parameters.mu).sum  / samples.size
    // estimatedMean: Double = -5.791574550006766
    println("estimated mean is " + estimatedMean)
    // estimated mean is -5.791574550006766

    val estimatedSigma = samples.map(sample => sample.parameters.sigma).sum / samples.size
    // estimatedSigma: Double = 17.350744030639415
    println("estimated sigma is " + estimatedSigma)
    // estimated sigma is 17.350744030639415


    val logger = new Logger()
    val mhIteratorWithLogging = chain.iterator(initialSample)
    val samples2 = mhIteratorWithLogging.drop(1000).take(3000).toIndexedSeq
    println("acceptance ratio is " +logger.acceptanceRatios())
*/

  }

}
