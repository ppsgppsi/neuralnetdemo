using System;

namespace NeuralNetDemo
{
    using System.Collections.Generic;
    using System.Linq;

    using Networks;

    public class DemoFileReader : ITrainingDataReader
    {
        private readonly List<string[]> lines;

        private int readIndex;

        public DemoFileReader(string filename)
        {
            lines = this.Parse(filename);
            readIndex = 0;
        }

        public int RecordCount
        {
            get
            {
                return lines.Count;
            }
        }

        public string[] NextRecord()
        {
            if (null == this.lines || (this.readIndex >= this.lines.Count))
            {
                return null;
            }

            return this.lines[readIndex++];
        }

        public void Reset()
        {
            readIndex = 0;
        }

        private List<string[]> Parse(string filename)
        {
            var ret = new List<string[]>();

            var fileLines = System.IO.File.ReadLines(filename);

            foreach (var fileLine in fileLines)
            {
                char[] sep = { ',' };
                var record = fileLine.Split(sep, StringSplitOptions.RemoveEmptyEntries).Select(s => s.Trim()).ToArray();

                if (record.Length > 2)
                {
                    ret.Add(record);
                }
            }
            return ret;
        }
    }
}
