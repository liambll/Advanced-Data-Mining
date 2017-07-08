# Data-Mining
## Project: Sentiment Analysis on Movie Comment
* Sentiment Analysis on 6,000 movie comments using Convolutional Neural Network 
* Technologies: Matlab, Matconvnet, GLoVE
-	read_data.m: read input and build vocabulary
-	vector_reprensentation.m: code to read GLoVE word embedding. It requires GLoVE files for corresponding word vector dimensions of 50, 100, 200 and 300: These files can be downloaded and extracted from GLoVE website.
-	train_model_experiment.m: code with cross validation to experiment with different model parameters
-	train_model_final.m: code to train final model using chosen model parameters and keep retrain word vectors
-	apply_model.m: apply the final model on a test set and write output to a file.

