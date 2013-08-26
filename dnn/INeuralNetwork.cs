namespace dnn
{
    interface INeuralNetwork
    {
        double Accuracy(double[][] data);        
        string ToString();
        void Train(double[][] data);
    }
}
