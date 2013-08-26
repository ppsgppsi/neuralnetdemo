using System;

namespace dnn
{
    public class ParticleProperties
    {
        public double V { get; set; }
        public double VSelf { get; set; }
        public double VSocial { get; set; }
        public double MaxVDelta { get; set; }
        public double MinVDelta { get; set; }
    }

    public class PsoParticle
    {
        private double bestAccuracy = 0.0;
        private readonly double[] velocities;

        private ParticleProperties props;

        private readonly Random rnd;

        public PsoParticle(Dnn dnn, ParticleProperties props, Random rnd)
        {
            this.Network = dnn;
            var data = dnn.Data;
            var numvelocities = /*weights: */(data.Props.NumInput * data.Props.NumHidden) + (data.Props.NumHidden * data.Props.NumOutput) /*bias: */ + data.Props.NumHidden + data.Props.NumOutput;
            velocities = new double[numvelocities];
            Array.Clear(velocities, 0, numvelocities);
            this.props = props;
            this.rnd = rnd;
        }

        private Dnn Network { get; set; }
        public Dnn Best { get; private set; }

        public void MoveTowards(Dnn socialbest)
        {
            if (null == socialbest)
            {
                throw new ArgumentNullException("socialbest");
            }

            if (this.Best == null)
            {
                this.Best = this.Network.Clone();
            }

            var data = this.Network.Data;
            var bestData = this.Best.Data;
            var socialBestData = socialbest.Data;
            var k = 0;

            this.SwarmMulti(data.ihWeights, bestData.ihWeights, socialBestData.ihWeights, ref k);
            this.Swarm(data.hBiases, bestData.hBiases, socialBestData.hBiases, ref k);
            this.SwarmMulti(data.hoWeights, bestData.hoWeights, socialBestData.hoWeights, ref k);
            this.Swarm(data.oBiases, bestData.oBiases, socialBestData.oBiases, ref k);
        }

        private void SwarmMulti(double[][] current, double[][] currentBest, double[][] socialBest, ref int k)
        {
            var rows = current.Length;

            for (int i = 0; i < rows; i++)
            {
                this.Swarm(current[i], currentBest[i], socialBest[i], ref k);
            }
        }

        private void Swarm(double[] current, double[] currentBest, double[] socialBest, ref int k)
        {
            var len = current.Length;

            for (int i = 0; i < len; i++)
            {
                //calculate new velocity => vfactor*v + vfactorP*random(0,1)*(best - p) + vfactorG*random(0,1)*(socialBest - p) 
                var velocity = this.props.V * this.velocities[k]
                               + (this.props.VSelf * rnd.NextDouble() * (currentBest[i] - current[i]))
                               + (this.props.VSocial * rnd.NextDouble() * (socialBest[i] - current[i]));

                if (velocity > this.props.MaxVDelta)
                {
                    velocity = this.props.MaxVDelta;
                }
                else if (velocity < this.props.MinVDelta)
                {
                    velocity = this.props.MinVDelta;
                }

                //calculate new position => p + v
                this.velocities[k++] = velocity;
                current[i] += velocity;
            }
        }

        public double UpdateProgress(double[][] testData)
        {
            var accuracy = this.Network.Accuracy(testData);

            if (accuracy > this.bestAccuracy)
            {
                this.bestAccuracy = accuracy;
                this.Best = this.Network.Clone();
            }

            return accuracy;
        }
    }
}
