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



namespace NeuralNetDemo
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Text;

    using Networks;

    public class NeuralNetDemo
    {
        private static void Main()
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

            var allData = new TrainingData();
            allData.LoadData("irisdata.txt", 4, 3);
            
            var sb = new StringBuilder();
            ArrayFormatter.Matrix(sb, allData.Data, 6, 1, true, "\nFirst 6 rows of entire 150-item data set:");
            Console.WriteLine(sb.ToString());

            Console.WriteLine("Creating 80% training and 20% test data matrices");         
            allData.Split(0.8);
            
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TrainData, 5, 1, true, "\nFirst 5 rows of training data:");
            Console.WriteLine(sb.ToString());
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TestData, 3, 1, true, "\nFirst 3 rows of test data:");
            Console.WriteLine(sb.ToString());

            allData.NormalizeInputs();            
            
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TrainData, 5, 1, true, "\nFirst 5 rows of normalized training data:");
            Console.WriteLine(sb.ToString());          
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TestData, 3, 1, true, "First 3 rows of normalized test data:");
            Console.Write(sb.ToString());

            Console.WriteLine("\nBuilding Neural Networks");
            Console.WriteLine("Hard-coded tanh function for input-to-hidden and softmax for hidden-to-output activations");

            var networks = new List<INeuralNetwork> { BuildBackPropNetwork(), BuildPsoNetwork() };

            foreach (var network in networks)
            {
                Console.WriteLine("\n\nTraining Network: " + network.GetType());
                var sw = new Stopwatch();
                sw.Start();
                network.Train(allData.TrainData);                
                var ts = sw.Elapsed;
                Console.WriteLine(
                    "Training complete in {0}",
                    string.Format("{0:00}:{1:00}:{2:00}.{3:000}", ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds));
                var trainAcc = network.Accuracy(allData.TrainData);
                var testAcc = network.Accuracy(allData.TestData);               
                Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));
                Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
                Console.WriteLine("\n\nFinal Network: {0}", network);
            }        

            Console.WriteLine("\nEnd Build 2013 neural network demo\n");
            Console.ReadLine();
        }      

        private static INeuralNetwork BuildPsoNetwork()
        {           
            var props = new NetworkProperties {
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
                                   NumNetworks = 4,                                   
                                   ParticleProps = particleProps
                               };

            return new PsoNetwork(netProps, props, new Random(0));                       
        }

        private static INeuralNetwork BuildBackPropNetwork()
        {                        
            var props = new NetworkProperties {
                InitWeightMin = -0.1,
                InitWeightMax = 0.1,
                NumHidden = 2,
                NumInput = 4,
                NumOutput = 3
            };

            var backProps = new BackPropProperties
                                {
                                    LearnRate = 0.05,
                                    MaxEprochs = 2000,
                                    Momentum = 0.00,
                                    WeightDecay = 0.000,
                                    MseStopCondition = 0.020
                                };
                  
            return  new BackPropNetwork(props, backProps, new Random(0));                                       
        }               
    } 
}

