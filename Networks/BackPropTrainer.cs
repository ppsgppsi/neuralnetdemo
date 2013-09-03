namespace Networks
{
    using System;
    using System.Text;

    public class BackPropProperties
    {
        public int MaxEprochs { get; set; }
        public double LearnRate { get; set; }
        public double Momentum { get; set; }
        public double WeightDecay { get; set; }
        public double MseStopCondition { get; set; }

        public BackPropProperties Clone()
        {
            return (BackPropProperties)this.MemberwiseClone();
        }
    }

    public class BackPropTrainer : INetworkTrainer
    {
        private readonly Random rnd;
        private readonly BackPropProperties backProps;

        // back-prop specific arrays (these could be local to method UpdateWeights)
        private readonly double[] oGrads; // output gradients for back-propagation
        private readonly double[] hGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays (could be local to method Train)
        private readonly double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
        private readonly double[] hPrevBiasesDelta;
        private readonly double[][] hoPrevWeightsDelta;
        private readonly double[] oPrevBiasesDelta;

        public BackPropTrainer(NeuralNetworkOptions ops, BackPropProperties backProps, Random rnd)
        {
            if (ops == null) { throw new ArgumentNullException("ops"); }
            if (backProps == null) { throw new ArgumentNullException("backProps");}

            this.backProps = backProps.Clone();
            this.Network = new NeuralNetwork(ops, rnd);
            this.rnd = rnd; // for Shuffle()

            var data = ops.DataProperties;

            // back-prop related arrays below
            this.hGrads = new double[data.NumHiddenNodes];
            this.oGrads = new double[data.NumOutputNodes];

            this.ihPrevWeightsDelta = NetworkData.MakeMatrix(data.NumInputNodes, data.NumHiddenNodes);
            this.hPrevBiasesDelta = new double[data.NumHiddenNodes];
            this.hoPrevWeightsDelta = NetworkData.MakeMatrix(data.NumHiddenNodes, data.NumOutputNodes);
            this.oPrevBiasesDelta = new double[data.NumOutputNodes];
        }

        public NeuralNetwork Network { get; private set; }

        public double Accuracy(double[][] data)
        {
            return this.Network.Accuracy(data);
        }     

        public override string ToString() // yikes
        {
            var s = this.Network.ToString();
            var sb = new StringBuilder(s);

            ArrayFormatter.Vector(sb, this.hGrads, 0, 4, true, "hGrads:");
            ArrayFormatter.Vector(sb, this.oGrads, 0, 4, true, "oGrads:");
            ArrayFormatter.Matrix(sb, this.ihPrevWeightsDelta, this.ihPrevWeightsDelta.Length, 4, true, "ihPrevWeightsDelta:");
            ArrayFormatter.Vector(sb, this.hPrevBiasesDelta, 0, 4, true,"hPrevBiasesDelta:");
            ArrayFormatter.Matrix(sb, this.hoPrevWeightsDelta, this.hoPrevWeightsDelta.Length, 4, true, "hoPrevWeightsDelta:");
            ArrayFormatter.Vector(sb, this.oPrevBiasesDelta, 0, 4, true, "oPrevBiasesDelta:");

            return sb.ToString();
        }

        public void Train(double[][] trainData)
        {
            var props = this.Network.Data.Props;
            // train a back-prop style NN classifier using learning rate and momentum
            // weight decay reduces the magnitude of a weight value over time unless that value
            // is constantly increased
            int epoch = 0;
            double[] tValues = new double[props.NumOutputNodes]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;


            while (epoch < this.backProps.MaxEprochs)
            {
                double mse = this.MeanSquaredError(trainData);
                if (mse < this.backProps.MseStopCondition) break; // consider passing value in as parameter
                //if (mse < 0.001) break; // consider passing value in as parameter

                Shuffle(this.rnd, sequence); // visit each training data in random order
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], props.NumInputNodes, tValues, 0, props.NumOutputNodes);
                    this.Network.ComputeOutputs(trainData[idx]); // copy xValues in, compute outputs (store them internally)
                    this.UpdateWeights(tValues, this.backProps.LearnRate, this.backProps.Momentum, this.backProps.WeightDecay); // find better weights
                } // each training tuple
                ++epoch;
            }
        }

        private void UpdateWeights(double[] tValues, double learnRate, double momentum, double weightDecay)
        {
            var data = this.Network.Data;
            var props = data.Props;
            // update the weights and biases using back-propagation, with target values, eta (learning rate),
            // alpha (momentum).
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
            // and matrices have values (other than 0.0)
            if (tValues.Length != props.NumOutputNodes)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < this.oGrads.Length; ++i)
            {
                // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - data.outputs[i]) * data.outputs[i];
                // 'mean squared error version' includes (1-y)(y) derivative
                this.oGrads[i] = derivative * (tValues[i] - data.outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < this.hGrads.Length; ++i)
            {
                // derivative of tanh = (1 - y) * (1 + y)
                double derivative = (1 - data.hOutputs[i]) * (1 + data.hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < props.NumOutputNodes; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = this.oGrads[j] * data.hoWeights[i][j];
                    sum += x;
                }
                this.hGrads[i] = derivative * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (int i = 0; i < data.ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < data.ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * this.hGrads[j] * data.inputs[i]; // compute the new delta
                    data.ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    data.ihWeights[i][j] += momentum * this.ihPrevWeightsDelta[i][j];
                    data.ihWeights[i][j] -= (weightDecay * data.ihWeights[i][j]); // weight decay
                    this.ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (int i = 0; i < data.hBiases.Length; ++i)
            {
                double delta = learnRate * this.hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                data.hBiases[i] += delta;
                data.hBiases[i] += momentum * this.hPrevBiasesDelta[i]; // momentum
                data.hBiases[i] -= (weightDecay * data.hBiases[i]); // weight decay
                this.hPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (int i = 0; i < data.hoWeights.Length; ++i)
            {
                for (int j = 0; j < data.hoWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * this.oGrads[j] * data.hOutputs[i];
                    data.hoWeights[i][j] += delta;
                    data.hoWeights[i][j] += momentum * this.hoPrevWeightsDelta[i][j]; // momentum
                    data.hoWeights[i][j] -= (weightDecay * data.hoWeights[i][j]); // weight decay
                    this.hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < data.oBiases.Length; ++i)
            {
                double delta = learnRate * this.oGrads[i] * 1.0;
                data.oBiases[i] += delta;
                data.oBiases[i] += momentum * this.oPrevBiasesDelta[i]; // momentum
                data.oBiases[i] -= (weightDecay * data.oBiases[i]); // weight decay
                this.oPrevBiasesDelta[i] = delta; // save
            }
        }        

        private static void Shuffle(Random rnd, int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
        {
            var data = this.Network.Data;
            var props = data.Props;

            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[props.NumInputNodes]; // first numInput values in trainData
            double[] tValues = new double[props.NumOutputNodes]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, props.NumInputNodes);
                Array.Copy(trainData[i], props.NumInputNodes, tValues, 0, props.NumOutputNodes); // get target values
                this.Network.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < props.NumOutputNodes; ++j)
                {
                    double err = tValues[j] - data.outputs[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }
    } 
}
