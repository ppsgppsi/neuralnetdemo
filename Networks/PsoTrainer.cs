namespace Networks
{
    using System;

    public class PsoNetworkProperties
    {
        public int NumNetworks { get; set; }
        public int Iterations { get; set; }
        public double DesiredAccuracy { get; set; }
        public ParticleProperties ParticleProps { get; set;} 
        
        public PsoNetworkProperties Clone()
        {
            var ret = (PsoNetworkProperties)this.MemberwiseClone();
            ret.ParticleProps = this.ParticleProps.Clone();
            return ret;
        }
    }

    public class PsoTrainer : INetworkTrainer
    {
        public PsoNetworkProperties PsoProps { get; private set; }
        public NeuralNetworkOptions NetworkOptions { get; private set; }
        private readonly Random rng;

        public PsoTrainer(NeuralNetworkOptions ops, PsoNetworkProperties netProps, Random rng)
        {
            if (ops == null) throw new ArgumentNullException("ops");
            if (netProps == null) throw new ArgumentNullException("netProps");
            if (rng == null) throw new ArgumentNullException("rng");

            this.PsoProps = netProps.Clone();
            this.NetworkOptions = ops.Clone();
            this.rng = rng;
        }
        public void Train(double[][] trainData)
        {           
            var particles = new PsoParticle[this.PsoProps.NumNetworks];

            for (int i = 0; i < this.PsoProps.NumNetworks; i++)
            {                
                particles[i] = new PsoParticle(new NeuralNetwork(this.NetworkOptions, this.rng), this.PsoProps.ParticleProps, this.rng);
            }           
                      
            var bestAccuracy = 0.0;           

            for (int i = 0; i < this.PsoProps.Iterations; i++)
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

        public override string ToString()
        {
            return this.Network.ToString();           
        }
    }
}
