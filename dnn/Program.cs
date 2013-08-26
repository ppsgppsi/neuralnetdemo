using System;

// For 2013 Microsoft Build Conference attendees
// June 25-28, 2013
// San Francisco, CA
//
// This is source for a C# console application.
// To compile you can 1.) create a new Visual Studio
// C# console app project named BuildNeuralNetworkDemo
// then zap away the template code and replace with this code,
// or 2.) copy this code into notepad, save as NeuralNetworkProgram.cs
// on your local machine, launch the special VS command shell
// (it knows where the csc.exe compiler is), cd-navigate to
// the directory containing the .cs file, type 'csc.exe
// NeuralNetworkProgram.cs' and hit enter, and then after 
// the compiler creates NeuralNetworkProgram.exe, you can
// run from the command line.
//
// This is an enhanced neural network. It is fully-connected
// and feed-forward. The training algorithm is back-propagation
// with momentum and weight decay. The inpput data is normalized
// so training is quite fast.
//
// You can use this code however you wish subject to the usual disclaimers
// (use at your own risk, etc.)



namespace dnn
{
    using System.Linq;  

    class NeuralNetworkProgram
    {
        static void Main()
        {
            Console.WriteLine("\nBegin Build 2013 neural network demo");
            Console.WriteLine("\nData is the famous Iris flower set.");
            Console.WriteLine("Data is sepal length, sepal width, petal length, petal width -> iris species");
            Console.WriteLine("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ");
            Console.WriteLine("The goal is to predict species from sepal length, width, petal length, width\n");

            Console.WriteLine("Raw data resembles:\n");
            Console.WriteLine(" 5.1, 3.5, 1.4, 0.2, Iris setosa");
            Console.WriteLine(" 7.0, 3.2, 4.7, 1.4, Iris versicolor");
            Console.WriteLine(" 6.3, 3.3, 6.0, 2.5, Iris virginica");
            Console.WriteLine(" ......\n");

            var sdata = System.IO.File.ReadAllText("irisdata.txt");
            var svalues = sdata.Split(new []{',','\n','\r'}, StringSplitOptions.RemoveEmptyEntries).Select(sValue => sValue.Trim()).ToArray();

            var allData = new double[svalues.Length/7][];

            var i = 0;           
            while (i < svalues.Length)
            {
                var row = new double[7];
                allData[i / 7] = row;

                for (var j = 0; j < 7; j++)
                {
                    row[j] = double.Parse(svalues[i++]);
                }              
            }          

            Console.WriteLine("\nFirst 6 rows of entire 150-item data set:");
            ShowMatrix(allData, 6, 1, true);

            Console.WriteLine("Creating 80% training and 20% test data matrices");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, out trainData, out testData);

            Console.WriteLine("\nFirst 5 rows of training data:");
            ShowMatrix(trainData, 5, 1, true);
            Console.WriteLine("First 3 rows of test data:");
            ShowMatrix(testData, 3, 1, true);

            Normalize(trainData, new int[] { 0, 1, 2, 3 });
            Normalize(testData, new int[] { 0, 1, 2, 3 });

            Console.WriteLine("\nFirst 5 rows of normalized training data:");
            ShowMatrix(trainData, 5, 1, true);
            Console.WriteLine("First 3 rows of normalized test data:");
            ShowMatrix(testData, 3, 1, true);

            RunBackPropDnn(trainData, testData);
            RunPsoDnn(trainData, testData);

            Console.WriteLine("\nEnd Build 2013 neural network demo\n");
            Console.ReadLine();

        } // Main

        static void RunPsoDnn(double[][] trainData, double[][] testData)
        {
            var rng = new Random(0);
            
            var props = new DnnProperties {
                              InitWeightMin = -0.1,
                              InitWeightMax = 0.1,
                              NumHidden = 2,
                              NumInput = 4,
                              NumOutput = 3
                          };

            var particleProps = new ParticleProperties {
                                        MaxVDelta = 2.0,
                                        MinVDelta = -2.0,
                                        V = 3.0,
                                        VSelf = 2.0,
                                        VSocial = 2.0
                                    };

            var netProps = new PsoNetworkProperties {
                                   DesiredAccuracy = 0.98,
                                   Iterations = 1000,
                                   NumNetworks = 11,                                   
                                   ParticleProps = particleProps
                               };

            var dnnPso = new PsoNetwork(netProps, props, rng);
            var network = dnnPso.Build(trainData);
            
            var trainAcc = network.Accuracy(trainData);
            var testAcc = network.Accuracy(testData);
            double[] weights = network.Data.GetWeights();
            Console.WriteLine("PSO Final neural network weights and bias values:");
            ShowVector(weights, 10, 3, true);
            Console.WriteLine("\nPSO Accuracy on training data = " + trainAcc.ToString("F4"));
            Console.WriteLine("\nPSO Accuracy on test data = " + testAcc.ToString("F4"));
            Console.WriteLine("\n\nFinal DNN: {0}", network);
        }

        static void RunBackPropDnn(double[][] trainData, double[][] testData)
        {
            Console.WriteLine("\nCreating a 4-input, 7-hidden, 3-output neural network");
            Console.Write("Hard-coded tanh function for input-to-hidden and softmax for ");
            Console.WriteLine("hidden-to-output activations");
            var props = new DnnProperties {
                InitWeightMin = -0.1,
                InitWeightMax = 0.1,
                NumHidden = 2,
                NumInput = 4,
                NumOutput = 3
            };
                  
            var nn = new BackPropNetwork(props, new Random(0));      

            int maxEpochs = 2000;
            double learnRate = 0.05;
            double momentum = 0.00;
            double weightDecay = 0.000;
            Console.WriteLine("Setting maxEpochs = 2000, learnRate = 0.05, momentum = 0.01, weightDecay = 0.0001");
            Console.WriteLine("Training has hard-coded mean squared error < 0.020 stopping condition");

            Console.WriteLine("\nBeginning training using incremental back-propagation\n");
            nn.Train(trainData, maxEpochs, learnRate, momentum, weightDecay);
            Console.WriteLine("Training complete");

            double[] weights = nn.Dnn.Data.GetWeights();
            Console.WriteLine("Final neural network weights and bias values:");
            ShowVector(weights, 10, 3, true);

            double trainAcc = nn.Dnn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            double testAcc = nn.Dnn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
        }

        static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
        {
            // split allData into 80% trainData and 20% testData
            Random rnd = new Random(0);
            int totRows = allData.Length;
            int numCols = allData[0].Length;

            int trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
            int testRows = totRows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            int[] sequence = new int[totRows]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                trainData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                testData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], testData[j], numCols);
                ++j;
            }
        } // MakeTrainTest

        static void Normalize(double[][] dataMatrix, int[] cols)
        {
            // normalize specified cols by computing (x - mean) / sd for each value
            foreach (int col in cols)
            {
                double sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][col];
                double mean = sum / dataMatrix.Length;
                sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean);
                // thanks to Dr. W. Winfrey, Concord Univ., for catching bug in original code
                double sd = Math.Sqrt(sum / (dataMatrix.Length - 1));
                for (int i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd;
            }
        }

        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-"); ;
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true) Console.WriteLine("");
        }

    } // class Program   
} // ns

