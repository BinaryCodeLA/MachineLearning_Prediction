using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Age_Detection
{
    public class AgeRange
    {
        [LoadColumn(0)]
        public string Name;

        [LoadColumn(1)]
        public float Age;

        [LoadColumn(2)]
        public string Gender;

        [LoadColumn(3)]
        public string Label;

    }
}
