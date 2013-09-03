using System;

namespace NeuralNetDemo
{
    using Networks;

    public class DemoFileReader : ITrainingDataReader
    {
        private readonly string[] lines;

        private int readIndex;

        public DemoFileReader(string filename)
        {
            lines = System.IO.File.ReadAllLines(filename);
            readIndex = 0;
        }

        public int RecordCount
        {
            get
            {
                return lines.Length;
            }
        }

        public string[] NextRecord()
        {
            if (null == lines || (readIndex >= lines.Length))
            {
                return null;
            }

            char[] sep = { ',' };

            return lines[readIndex++].Split(sep, StringSplitOptions.RemoveEmptyEntries);
        }
    }
}
