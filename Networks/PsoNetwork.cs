namespace Networks
{
    using System;

    public class PsoNetworkProperties
    {
        public int NumNetworks { get; set; }
        public int Iterations { get; set; }
        public double DesiredAccuracy { get; set; }
        public ParticleProperties ParticleProps { get; set;}                   
    }

    public class PsoNetwork : INeuralNetwork
    {
        public PsoNetworkProperties PsoProps { get; private set; }
        public NetworkProperties NetworkProps { get; private set; }
        private readonly Random rng;

        public PsoNetwork(PsoNetworkProperties netProps, NetworkProperties props, Random rng)
        {
            this.PsoProps = netProps;
            this.NetworkProps = props;
            this.rng = rng;
        }
        public void Train(double[][] trainData)
        {           
            var particles = new PsoParticle[this.PsoProps.NumNetworks];

            for (int i = 0; i < this.PsoProps.NumNetworks; i++)
            {                
                particles[i] = new PsoParticle(new NeuralNetwork(this.NetworkProps, this.rng), this.PsoProps.ParticleProps, this.rng);
            }           

            var foundNetwork = false;            
            var bestAccuracy = 0.0;           

            for (int i = 0; i < this.PsoProps.Iterations && !foundNetwork; i++)
            {                
                for (var p = 0; p < particles.Length; p++)
                {
                    var accuracy = particles[p].UpdatePersonalBest(trainData);
                    if (accuracy > bestAccuracy)
                    {
                        this.Network = particles[p].Best.Clone();                       
                        bestAccuracy = accuracy;                    
                    }
                    if (accuracy > this.PsoProps.DesiredAccuracy)
                    {
                        foundNetwork = true;
                        break;
                    }
                }

                for (var p = 0; p < particles.Length && !foundNetwork; p++)
                {
                    particles[p].MoveTowards(this.Network);
                }
            }            
        }

        public NeuralNetwork Network { get; private set; }

        public double Accuracy(double[][] data)
        {
            return this.Network.Accuracy(data);
        }

        public override string ToString() // yikes
        {
            return this.Network.ToString();           
        }
    }
}
