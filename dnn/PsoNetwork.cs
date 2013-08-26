namespace dnn
{
    using System;
    using System.Text;

    public class PsoNetworkProperties
    {
        public int NumNetworks { get; set; }
        public int Iterations { get; set; }
        public double DesiredAccuracy { get; set; }
        public ParticleProperties ParticleProps { get; set;}                   
    }

    public class PsoNetwork : INeuralNetwork
    {
        public PsoNetworkProperties NetworkProps { get; private set; }
        public DnnProperties DnnProps { get; private set; }
        private readonly Random rng;

        public PsoNetwork(PsoNetworkProperties netProps, DnnProperties props, Random rng)
        {
            this.NetworkProps = netProps;
            this.DnnProps = props;
            this.rng = rng;
        }
        public void Train(double[][] trainData)
        {           
            var particles = new PsoParticle[NetworkProps.NumNetworks];

            for (int i = 0; i < NetworkProps.NumNetworks; i++)
            {                
                particles[i] = new PsoParticle(new Dnn(this.DnnProps, this.rng), NetworkProps.ParticleProps, this.rng);
            }           

            var foundNetwork = false;            
            var bestAccuracy = 0.0;

            foreach (var particle in particles)
            {
                var accuracy = particle.UpdatePersonalBest(trainData);

                if (accuracy > bestAccuracy)
                {
                    this.Network = particle.Best.Clone();                   
                    bestAccuracy = accuracy;
                }
            }

            for (int i = 0; i < NetworkProps.Iterations && !foundNetwork; i++)
            {
                foreach (var particle in particles)
                {
                    particle.MoveTowards(this.Network);
                }

                for (var p = 0; p < particles.Length; p++)
                {
                    var accuracy = particles[p].UpdatePersonalBest(trainData);
                    if (accuracy > bestAccuracy)
                    {
                        this.Network = particles[p].Best.Clone();                       
                        bestAccuracy = accuracy;                    
                    }
                    if (accuracy > NetworkProps.DesiredAccuracy)
                    {
                        foundNetwork = true;
                        break;
                    }
                }                           
            }            
        }

        public Dnn Network { get; private set; }

        public double Accuracy(double[][] data)
        {
            return Network.Accuracy(data);
        }

        public override string ToString() // yikes
        {
            return this.Network.ToString();           
        }
    }
}
