using System;

// Based on original code by Dr. James Mccaffrey. See LICENSE.txt for additional information.

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
            Console.WriteLine(" 5.1, 3.5, 1.4, 0.2, setosa");
            Console.WriteLine(" 7.0, 3.2, 4.7, 1.4, versicolor");
            Console.WriteLine(" 6.3, 3.3, 6.0, 2.5, virginica");
            Console.WriteLine(" ......\n");

            var reader = new DemoFileReader("irisdata.txt");            

            var inputEncoders = new INeuralDataEncoder[4];

            for (int i = 0; i < inputEncoders.Length; ++i)
            {
                inputEncoders[i] = new GaussianNormalizer(reader.RecordCount);
            }
            var outputEncoder = DataEncodeDecodeFactory.CreateDataEncoder(props.Output, reader.RecordCount);

            var allData = new TrainingData();
            allData.LoadData(reader, inputEncoders, outputEncoder);
            
            var sb = new StringBuilder();

            Console.WriteLine("Creating 80% training and 20% test data matrices");         
            allData.Split(0.8);
            
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TrainData, 5, 1, true, "\nFirst 5 rows of normalized training data:");
            Console.WriteLine(sb.ToString());          
            sb.Clear();
            ArrayFormatter.Matrix(sb, allData.TestData, 3, 1, true, "First 3 rows of normalized test data:");
            Console.Write(sb.ToString());

            Console.WriteLine("\nBuilding Neural Networks");
            Console.WriteLine("Hard-coded tanh function for input-to-hidden and softmax for hidden-to-output activations");

            var networks = new List<INetworkTrainer> { BuildBackPropNetwork(), BuildPsoNetwork() };

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

        private static INetworkTrainer BuildPsoNetwork()
        {           
            var props = new NetworkDataProperties {
                              InitWeightMin = -0.1,
                              InitWeightMax = 0.1,
                              NumHiddenNodes = 2,
                              NumInputNodes = 4,
                              NumOutputNodes = 3
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

            return new PsoTrainer(netProps, props, new Random(0));                       
        }

        private static INetworkTrainer BuildBackPropNetwork()
        {                        
            var props = new NetworkDataProperties {
                InitWeightMin = -0.1,
                InitWeightMax = 0.1,
                NumHiddenNodes = 2,
                NumInputNodes = 4,
                NumOutputNodes = 3
            };

            var backProps = new BackPropProperties
                                {
                                    LearnRate = 0.05,
                                    MaxEprochs = 2000,
                                    Momentum = 0.00,
                                    WeightDecay = 0.000,
                                    MseStopCondition = 0.020
                                };
                  
            return  new BackPropTrainer(props, backProps, new Random(0));                                       
        }               
    } 
}

