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

            double pctComplete = this.PsoProps.Iterations / 10;

            var bestAccuracy = 0.0;           

            for (int i = 0; i < this.PsoProps.Iterations; i++)
            {
                if ((i + 1) % pctComplete == 0)
                    Console.Write((100 * (i + 1) / this.PsoProps.Iterations).ToString() + @"%, ");
                //if (((i+1) % 100) == 0)
                //    Console.Write((100 * (i+1) / this.PsoProps.Iterations).ToString() + @"%, ");

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
                        Console.WriteLine((100 * (i + 1) / this.PsoProps.Iterations).ToString() + @"% (training ended)");
                        return;
                    }
                }

                for (var p = 0; p < particles.Length; p++)
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
