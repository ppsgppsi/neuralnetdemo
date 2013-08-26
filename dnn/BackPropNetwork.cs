namespace dnn
{
    using System;
    using System.Text;

    public class BackPropNetwork
    {
        private Random rnd;

        // back-prop specific arrays (these could be local to method UpdateWeights)
        private double[] oGrads; // output gradients for back-propagation
        private double[] hGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays (could be local to method Train)
        private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;


        public BackPropNetwork(DnnProperties props, Random rnd)
        {
            this.rnd = rnd; // for Shuffle()

            this.Dnn = new Dnn(props, rnd);

            // back-prop related arrays below
            this.hGrads = new double[props.NumHidden];
            this.oGrads = new double[props.NumOutput];

            this.ihPrevWeightsDelta = DnnData.MakeMatrix(props.NumInput, props.NumHidden);
            this.hPrevBiasesDelta = new double[props.NumHidden];
            this.hoPrevWeightsDelta = DnnData.MakeMatrix(props.NumHidden, props.NumOutput);
            this.oPrevBiasesDelta = new double[props.NumOutput];
        }

        public Dnn Dnn { get; private set; }

        public override string ToString() // yikes
        {
            var s = this.Dnn.ToString();
            var sb = new StringBuilder(s);

            DnnData.ArrayToString(sb, "F4", "hGrads:", hGrads);
            DnnData.ArrayToString(sb, "F4", "oGrads:", oGrads);
            DnnData.MatrixToString(sb, "ihPrevWeightsDelta:", ihPrevWeightsDelta);
            DnnData.ArrayToString(sb, "F4", "hPrevBiasesDelta:", hPrevBiasesDelta);
            DnnData.MatrixToString(sb, "hoPrevWeightsDelta:", hoPrevWeightsDelta);
            DnnData.ArrayToString(sb, "F4", "oPrevBiasesDelta:", oPrevBiasesDelta);

            return sb.ToString();
        }

        private void UpdateWeights(double[] tValues, double learnRate, double momentum, double weightDecay)
        {
            var data = this.Dnn.Data;
            var props = data.Props;
            // update the weights and biases using back-propagation, with target values, eta (learning rate),
            // alpha (momentum).
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
            // and matrices have values (other than 0.0)
            if (tValues.Length != props.NumOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < oGrads.Length; ++i)
            {
                // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - data.outputs[i]) * data.outputs[i];
                // 'mean squared error version' includes (1-y)(y) derivative
                oGrads[i] = derivative * (tValues[i] - data.outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < hGrads.Length; ++i)
            {
                // derivative of tanh = (1 - y) * (1 + y)
                double derivative = (1 - data.hOutputs[i]) * (1 + data.hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < props.NumOutput; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = oGrads[j] * data.hoWeights[i][j];
                    sum += x;
                }
                hGrads[i] = derivative * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (int i = 0; i < data.ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < data.ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * hGrads[j] * data.inputs[i]; // compute the new delta
                    data.ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                    // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    data.ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    data.ihWeights[i][j] -= (weightDecay * data.ihWeights[i][j]); // weight decay
                    ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (int i = 0; i < data.hBiases.Length; ++i)
            {
                double delta = learnRate * hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                data.hBiases[i] += delta;
                data.hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
                data.hBiases[i] -= (weightDecay * data.hBiases[i]); // weight decay
                hPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (int i = 0; i < data.hoWeights.Length; ++i)
            {
                for (int j = 0; j < data.hoWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * oGrads[j] * data.hOutputs[i];
                    data.hoWeights[i][j] += delta;
                    data.hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
                    data.hoWeights[i][j] -= (weightDecay * data.hoWeights[i][j]); // weight decay
                    hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < data.oBiases.Length; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                data.oBiases[i] += delta;
                data.oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
                data.oBiases[i] -= (weightDecay * data.oBiases[i]); // weight decay
                oPrevBiasesDelta[i] = delta; // save
            }
        } // UpdateWeights

        // ----------------------------------------------------------------------------------------

        public void Train(double[][] trainData, int maxEprochs, double learnRate, double momentum,
              double weightDecay)
        {
            var props = this.Dnn.Data.Props;
            // train a back-prop style NN classifier using learning rate and momentum
            // weight decay reduces the magnitude of a weight value over time unless that value
            // is constantly increased
            int epoch = 0;
            double[] tValues = new double[props.NumOutput]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;


            while (epoch < maxEprochs)
            {
                double mse = MeanSquaredError(trainData);
                if (mse < 0.020) break; // consider passing value in as parameter
                //if (mse < 0.001) break; // consider passing value in as parameter

                Shuffle(this.rnd, sequence); // visit each training data in random order
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], props.NumInput, tValues, 0, props.NumOutput);
                    this.Dnn.ComputeOutputs(trainData[idx]); // copy xValues in, compute outputs (store them internally)
                    UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
                } // each training tuple
                ++epoch;
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
            var data = this.Dnn.Data;
            var props = data.Props;

            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[props.NumInput]; // first numInput values in trainData
            double[] tValues = new double[props.NumOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, props.NumInput);
                Array.Copy(trainData[i], props.NumInput, tValues, 0, props.NumOutput); // get target values
                this.Dnn.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < props.NumOutput; ++j)
                {
                    double err = tValues[j] - data.outputs[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainData.Length;
        }

    } // NeuralNetwork
}
