namespace Networks
{
    public interface INetworkTrainer
    {
        double Accuracy(double[][] data); 
               
        void Train(double[][] trainingData);
    }
}
