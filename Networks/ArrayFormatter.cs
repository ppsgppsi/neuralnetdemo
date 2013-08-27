namespace Networks
{
    using System;
    using System.Globalization;
    using System.Text;

    public static class ArrayFormatter
    {              
        public static void Vector(StringBuilder sb, double[] vector, int valsPerRow, int decimals, bool newLine, string header)
        {
            if (header != null)
            {
                sb.Append(header).Append("\n");
            }

            for (int i = 0; i < vector.Length; ++i)
            {
                if ((valsPerRow > 0) && (i % valsPerRow == 0))
                {
                    sb.Append("\n");
                }
                sb.Append(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine)
            {
                sb.Append("\n");
            }
        }

        public static void Matrix(StringBuilder sb, double[][] matrix, int numRows, int decimals, bool newLine, string header)
        {
            if (header != null)
            {
                sb.Append(header).Append("\n"); 
            }

            for (int i = 0; i < numRows; ++i)
            {
                sb.Append(i.ToString(CultureInfo.InvariantCulture).PadLeft(3) + ": ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    sb.Append(matrix[i][j] >= 0.0 ? " " : "-");
                    sb.Append(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                sb.Append("\n");
            }
            if (newLine)
            {
                sb.Append("\n");
            }
        }
    }
}
