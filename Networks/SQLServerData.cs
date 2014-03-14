using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Data;
using System.Data.SqlClient;
using System.Configuration;

namespace Networks
{
    class SQLServerData
    {
        private string connectionString = @"server=.\SQL2012;;Uid=;Pwd=;Database=Stock;Integrated Security=SSPI";
        private DataSet _ds = new DataSet();

        public double[][] data_array;


        public void GetDataset()
        {

            using (SqlConnection conn = new SqlConnection(connectionString))
            {
                conn.Open();
                using (SqlDataAdapter cmd = new SqlDataAdapter("up_NN_TrainDataGet", conn))
                {
                    cmd.SelectCommand.CommandType = CommandType.StoredProcedure;

                    var param = cmd.SelectCommand.Parameters;
                    //param.Add("@Test", SqlDbType.VarChar, -1).Value = "abc";

                    cmd.Fill(_ds);
                    cmd.Dispose();
                }
                conn.Close();
            }

            AssignToArray();

        }

        void AssignToArray()
        {

            DataTable dt = _ds.Tables[0];
            int rowCount = dt.Rows.Count;

            data_array = new double[rowCount][];

            int columnCount = dt.Columns.Count;
            int aint = 0;

            for (int i = 0; i < rowCount; i++)
            {
                DataRow dr = dt.Rows[i];

                double[] dim2_array = new double[columnCount];

                for (int c = 0; c < columnCount; c++)
                    dim2_array[c] = Convert.ToDouble(dr[c]);

                data_array[aint++] = dim2_array;
            }
        }

    }

}
