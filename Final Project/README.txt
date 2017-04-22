Final Project
Team Meambers:  Rajarshi Biswas
				Sayam Ganguly
How to run - 
Step 1 : Run the file "Preprocessing.py". It will generate 11 Matlab p-code file that will be used by subsequent programs.
		 It will generate the following 11 files - 
			1. all_set.p
			2. training_set.p
			3. testing_set.p
			4. train_b.p
			5. train_t.p
			6. train_e.p
			7. train_m.p
			8. test_b.p
			9. test_t.p
			10. test_e.p
			11. test_m.p
Step 2 : Run the following files to run classification
			1. BagOfWords_Model.py
			2. TfIdf_Model.py
			3. BiGram_Model.py
			4. TriGram_Model.py
		Each classification model file will output corresonding reults obtained from classification in text files. Following are the text files - 
			1. bagofwords_results.txt
			2. tfidf_results.txt
			3. bigram_results.txt
			4. trigram_results.txt
		In addition this step will generate 4 files which are the saved models and vectorizers to be used for twitter prediction
			1. BOW_MNB_Model.sav
			2. BOW_MNB_Model.pk
			3. TFIDF_LRG_Model.sav
			4. TFIDF_LRG_Model.pk
Step 3 : Run Twitter_Predict.py to predict genre of live twitter stream by previously saved models. It will generate a csv file with the prediction details - 
			1. Twitter_Prediction_Result.csv