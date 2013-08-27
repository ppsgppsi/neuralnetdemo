namespace Networks
{
    public interface INeuralNetwork
    {
        double Accuracy(double[][] data); 
               
        void Train(double[][] data);
    }
}
