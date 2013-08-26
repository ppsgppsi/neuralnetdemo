using System;
using System.Linq;

namespace dnn
{   
    public class DnnData
    {        
        public DnnProperties Props { get; private set; }       

        // ReSharper disable InconsistentNaming
        
        //The input variables that run this DNN
        public readonly double[][] ihWeights; // input-hidden
        public readonly double[] hBiases;
        public readonly double[][] hoWeights; // hidden-output
        public readonly double[] oBiases;

        //The values of each node in the DNN, inputs are given, outputs are computed.
        //Stored here for performance and historical reasons only, not required to run the DNN.
        public readonly double[] inputs;
        public readonly double[] hOutputs;
        public readonly double[] outputs;

        // ReSharper restore InconsistentNaming

        public DnnData(DnnProperties props)
        {
            this.Props = props.Clone();        
            
            //weights
            this.ihWeights = MakeMatrix(Props.NumInput, Props.NumHidden);
            this.hBiases = new double[Props.NumHidden];            
            this.hoWeights = MakeMatrix(Props.NumHidden, Props.NumOutput);
            this.oBiases = new double[Props.NumOutput];

            //input and output node values
            this.hOutputs = new double[Props.NumHidden];
            this.inputs = new double[Props.NumInput];
            this.outputs = new double[Props.NumOutput];            
        }

        public bool IsEqual(DnnData other)
        {
            if (this.Props.NumInput != other.Props.NumInput) return false;
            if (this.Props.NumHidden != other.Props.NumHidden) return false;
            if (this.Props.NumOutput != other.Props.NumOutput) return false;

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

        public DnnData Clone()
        {
            var ret = new DnnData(Props);

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
            int numWeights = (Props.NumInput * Props.NumHidden) + (Props.NumHidden * Props.NumOutput) + Props.NumHidden + Props.NumOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < Props.NumInput; ++i)
                for (int j = 0; j < Props.NumHidden; ++j)
                    this.ihWeights[i][j] = weights[k++];
            for (int i = 0; i < Props.NumHidden; ++i)
                this.hBiases[i] = weights[k++];
            for (int i = 0; i < Props.NumHidden; ++i)
                for (int j = 0; j < Props.NumOutput; ++j)
                    this.hoWeights[i][j] = weights[k++];
            for (int i = 0; i < Props.NumOutput; ++i)
                this.oBiases[i] = weights[k++];
        }

        public void InitializeWeights(Random rnd, double lo, double hi)
        {
            // initialize weights and biases to small random values
            int numWeights = (Props.NumInput * Props.NumHidden) + (Props.NumHidden * Props.NumOutput) + Props.NumHidden + Props.NumOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }

        public double[] GetWeights()
        {
            // returns the current set of wweights, presumably after training
            int numWeights = (Props.NumInput * Props.NumHidden) + (Props.NumHidden * Props.NumOutput) + Props.NumHidden + Props.NumOutput;
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
    }
}
