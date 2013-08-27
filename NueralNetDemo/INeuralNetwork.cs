namespace nueraldemo
{
    interface INeuralNetwork
    {
        double Accuracy(double[][] data);                
        void Train(double[][] data);
    }
}
