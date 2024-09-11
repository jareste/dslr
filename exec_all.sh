echo "All students data"
echo "-----------------------------------------------------------------------"

python srcs/describe.py datasets/dataset_train.csv

echo "-----------------------------------------------------------------------\n\n"
echo "Histogram"
echo "-----------------------------------------------------------------------"

python srcs/histogram.py datasets/dataset_train.csv --all

echo "-----------------------------------------------------------------------\n\n"
echo "Scatter plot"
echo "-----------------------------------------------------------------------"

python srcs/scatter_plot.py datasets/dataset_train.csv

echo "-----------------------------------------------------------------------\n\n"
echo "Pair plot launched on background..."
echo "-----------------------------------------------------------------------"

python srcs/pair_plot.py datasets/dataset_train.csv &

echo "-----------------------------------------------------------------------\n\n"
echo "logreg_train"
echo "-----------------------------------------------------------------------"

python srcs/logreg_train.py datasets/dataset_train.csv

echo "-----------------------------------------------------------------------\n\n"
echo "logreg_predict"
echo "-----------------------------------------------------------------------"

python srcs/logreg_predict.py datasets/dataset_test.csv
