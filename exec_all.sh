echo "All students data"
echo "-----------------------------------------------------------------------"

python srcs/describe.py datasets/dataset_train.csv

# echo "-----------------------------------------------------------------------\n\n\n\n"
# echo "All Houses data"
# echo "-----------------------------------------------------------------------"


# python srcs/describe.py datasets/dataset_train.csv --house

echo "-----------------------------------------------------------------------\n\n\n\n"
echo "Histogram"
echo "-----------------------------------------------------------------------"

python srcs/histogram.py datasets/dataset_train.csv --all

# echo "-----------------------------------------------------------------------\n\n\n\n"
# echo "Scatter plot"
# echo "-----------------------------------------------------------------------"

# python srcs/scatter_plot.py datasets/dataset_train.csv