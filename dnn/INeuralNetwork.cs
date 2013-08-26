namespace dnn
{
    interface INeuralNetwork
    {
        double Accuracy(double[][] data);
        string WeightsAsString();
        string ToString();
    }
}
