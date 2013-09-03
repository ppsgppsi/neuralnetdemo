namespace Networks
{
    using System;
    using System.Linq;
    using System.Text;

    public class NetworkDataProperties
    {
        public int NumInputNodes { get; set; }
        public int NumHiddenNodes { get; set; }
        public int NumOutputNodes { get; set; }
        public double InitWeightMin { get; set; }
        public double InitWeightMax { get; set; }

        public NetworkDataProperties Clone()
        {
            return this.MemberwiseClone() as NetworkDataProperties;
        }
    }

    public class NetworkData
    {        
        public NetworkDataProperties Props { get; private set; }       

        // ReSharper disable InconsistentNaming
        
        //The input variables that run this Neural Network
        public readonly double[][] ihWeights; // input-hidden
        public readonly double[] hBiases;
        public readonly double[][] hoWeights; // hidden-output
        public readonly double[] oBiases;

        //The values of each node in the Network, inputs are given, outputs are computed.
        //Stored here for performance and historical reasons only, not required to run the Neural Network.
        public readonly double[] inputs;
        public readonly double[] hOutputs;
        public readonly double[] outputs;

        // ReSharper restore InconsistentNaming

        public NetworkData(NetworkDataProperties props)
        {
            this.Props = props.Clone();        
            
            //weights
            this.ihWeights = MakeMatrix(this.Props.NumInputNodes, this.Props.NumHiddenNodes);
            this.hBiases = new double[this.Props.NumHiddenNodes];            
            this.hoWeights = MakeMatrix(this.Props.NumHiddenNodes, this.Props.NumOutputNodes);
            this.oBiases = new double[this.Props.NumOutputNodes];

            //input and output node values
            this.hOutputs = new double[this.Props.NumHiddenNodes];
            this.inputs = new double[this.Props.NumInputNodes];
            this.outputs = new double[this.Props.NumOutputNodes];            
        }

        public bool IsEqual(NetworkData other)
        {
            if (this.Props.NumInputNodes != other.Props.NumInputNodes) return false;
            if (this.Props.NumHiddenNodes != other.Props.NumHiddenNodes) return false;
            if (this.Props.NumOutputNodes != other.Props.NumOutputNodes) return false;

            if (!this.hBiases.SequenceEqual(other.hBiases)) return false;
            if (!this.oBiases.SequenceEqual(other.oBiases)) return false;

            for (var i = 0; i < this.ihWeights.Length; i++)
            {
                if (!this.ihWeights[i].SequenceEqual(other.ihWeights[i])) return false;
            }
            for (var i = 0; i < this.hoWeights.Length; i++)
            {
                if (!this.hoWeights[i].SequenceEqual(other.hoWeights[i])) return false;
            }
            return true;
        }

        public NetworkData Clone()
        {
            var ret = new NetworkData(this.Props);

            Buffer.BlockCopy(this.hBiases, 0, ret.hBiases, 0, Buffer.ByteLength(this.hBiases));
            Buffer.BlockCopy(this.oBiases, 0, ret.oBiases, 0, Buffer.ByteLength(this.oBiases));

            for (int i = 0; i < this.ihWeights.Length; i++)
            {
                Buffer.BlockCopy(this.ihWeights[i], 0, ret.ihWeights[i], 0, Buffer.ByteLength(this.ihWeights[0]));
            }

            for (int i = 0; i < this.hoWeights.Length; i++)
            {
                Buffer.BlockCopy(this.hoWeights[i], 0, ret.hoWeights[i], 0, Buffer.ByteLength(this.hoWeights[0]));
            }
            return ret;
        }

        public static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            var result = new double[rows][];
            for (var r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (this.Props.NumInputNodes * this.Props.NumHiddenNodes) + (this.Props.NumHiddenNodes * this.Props.NumOutputNodes) + this.Props.NumHiddenNodes + this.Props.NumOutputNodes;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < this.Props.NumInputNodes; ++i)
                for (int j = 0; j < this.Props.NumHiddenNodes; ++j)
                    this.ihWeights[i][j] = weights[k++];
            for (int i = 0; i < this.Props.NumHiddenNodes; ++i)
                this.hBiases[i] = weights[k++];
            for (int i = 0; i < this.Props.NumHiddenNodes; ++i)
                for (int j = 0; j < this.Props.NumOutputNodes; ++j)
                    this.hoWeights[i][j] = weights[k++];
            for (int i = 0; i < this.Props.NumOutputNodes; ++i)
                this.oBiases[i] = weights[k++];
        }

        public void InitializeWeights(Random rnd, double lo, double hi)
        {
            // initialize weights and biases to small random values
            int numWeights = (this.Props.NumInputNodes * this.Props.NumHiddenNodes) + (this.Props.NumHiddenNodes * this.Props.NumOutputNodes) + this.Props.NumHiddenNodes + this.Props.NumOutputNodes;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }

        public double[] Weights()
        {
            // returns the current set of wweights, presumably after training
            int numWeights = (this.Props.NumInputNodes * this.Props.NumHiddenNodes) + (this.Props.NumHiddenNodes * this.Props.NumOutputNodes) + this.Props.NumHiddenNodes + this.Props.NumOutputNodes;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < this.ihWeights.Length; ++i)
                for (int j = 0; j < this.ihWeights[0].Length; ++j)
                    result[k++] = this.ihWeights[i][j];
            for (int i = 0; i < this.hBiases.Length; ++i)
                result[k++] = this.hBiases[i];
            for (int i = 0; i < this.hoWeights.Length; ++i)
                for (int j = 0; j < this.hoWeights[0].Length; ++j)
                    result[k++] = this.hoWeights[i][j];
            for (int i = 0; i < this.oBiases.Length; ++i)
                result[k++] = this.oBiases[i];
            return result;
        }   

        public override string ToString()
        {
            var sb = new StringBuilder();

            sb.Append("numInput = " + this.Props.NumInputNodes + " numHidden = " + this.Props.NumHiddenNodes + " numOutput = " + this.Props.NumOutputNodes + "\n\n");
                   
            ArrayFormatter.Matrix(sb, this.ihWeights, this.ihWeights.Length, 4, true, "ihWeights:");
            ArrayFormatter.Vector(sb, this.hBiases, 0, 4, true, "hBiases:");                
            ArrayFormatter.Matrix(sb, this.hoWeights, this.hoWeights.Length, 4, true, "hoWeights:");
            ArrayFormatter.Vector(sb, this.oBiases, 0, 4, true, "hBiases:");           
            return sb.ToString();
        }        
    }
}
