from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from collections import defaultdict



def main():
	#########################################
	########## SETUP ENVIRONMENT ############
	#########################################

	##SET SEED
	random_state = 876539
	

	##GET DATA
	data = pd.read_csv('./data/ProsperLoanData.csv')


	##RENAME FIELDS
	data.rename(columns = {"ProsperRating (numeric)": "NumericProsperRating", "ProsperRating (Alpha)": "AlpaProsperRating", "ListingCategory (numeric)": "NumericListingCategory"}, inplace = True)


	##GET RELEVANT FIELDS
	data = data[["Term","BorrowerRate","NumericProsperRating","AlpaProsperRating","ProsperScore","NumericListingCategory","BorrowerState","Occupation",
				"EmploymentStatus","EmploymentStatusDuration","IsBorrowerHomeowner","CreditScoreRangeLower","CurrentCreditLines","OpenCreditLines",
				"TotalCreditLinespast7years","OpenRevolvingAccounts","OpenRevolvingMonthlyPayment","InquiriesLast6Months","TotalInquiries","CurrentDelinquencies","AmountDelinquent",
				"DelinquenciesLast7Years","PublicRecordsLast10Years","PublicRecordsLast12Months","RevolvingCreditBalance","BankcardUtilization","AvailableBankcardCredit",
				"DebtToIncomeRatio","IncomeRange","StatedMonthlyIncome","LoanOriginalAmount"]]


		
	####################################################################
	############## FEATURE PROCESSING AND ENGINEERNG ###################
	####################################################################
		
	##REMOVE SPARSE COLUMNS
	sparse_count = data.isnull().sum()
	sparse_percent = round(sparse_count/data.shape[0] * 100, 2)
	threshold = 20
	sparse_percent = sparse_percent.loc[sparse_percent >= threshold]
	data = data.drop(list(sparse_percent.index), axis = 1)

	
	##REMOVE ROWS WITH NA
	data = data.dropna(axis = 0)


	##REMOVE SPECIFIED FIELDS
	column_list = ["NumericProsperRating","AlpaProsperRating","ProsperScore"]
	if any(data.columns.isin(column_list)):
		data = data.drop(column_list, axis = 1)

		
	##CREATE CLASS FOR MULTI LABEL ENCODING 	
	class MultiLabelEncoder():
		columns = None
		encoders = defaultdict(LabelEncoder)

		def __init__(self, cols):
			self.columns = cols
			
		def fit(self, X, y=None):
			for col in self.columns:
				if any(X.columns.isin([col])):
					encoder = LabelEncoder()
					encoder.fit(X[col])
					X[col] =  encoder.transform(X[[col]])
					self.encoders[col] = encoder
			return self
			
		def transform(self,X, y = None):
			for col in self.columns:
				if any(X.columns.isin([col])):
					encoder = encoders[col]
					X[col] = encoder.transform(X[[col]])
			return X
			
		def fit_transform(self,X, y = None):
			print("INSIDE FIT_TRANSFORM")
			for col in self.columns:
				if any(X.columns.isin([col])):
					encoder = LabelEncoder()
					encoder.fit(X[col])
					X[col] =  encoder.transform(X[[col]])
					self.encoders[col] = encoder
			return X	
			
		def inverse_transform(self, X, y = None):
			for col in self.columns:
				if any(X.columns.isin([col])):
					encoder = encoders[col]
					X[col] = encoder.inverse_transform(X[[col]])
			return X

	##PREPROCESS CATEGORICAL DATA TO FACTORS (LABEL ENCODED)	
	cat_features = ["Term","BorrowerState", "Occupation", "EmploymentStatus", "IsBorrowerHomeowner", "IncomeRange","NumericListingCategory"]
	prep_multiLabelEncoding = MultiLabelEncoder(cat_features)
	data = prep_multiLabelEncoding.fit_transform(data)
			
			
	##SEPARATE X AND Y
	data_y = data["BorrowerRate"]
	data_X = data.drop(["BorrowerRate"], axis = 1)
	
	


	###############################################
	#### PERFORM RECURSIVE FEATURE ELIMINATION ####
	###############################################

	##CREATE VARIABLES
	random_state = 28655
	estimator = DecisionTreeRegressor(random_state = random_state)
	
	
	##PERFROM RFECV
	step = 1
	n_features = data_X.shape[1]
	min_features_to_select = 2
	rfecv = RFECV(estimator = estimator , n_jobs = 1, step = step, cv = 2, verbose = 1, scoring = "neg_mean_squared_error", min_features_to_select = min_features_to_select)
	rfecv.fit(data_X, data_y)
	print("RFECV RESULTS:")
	print(rfecv.grid_scores_)

	##STORE CHART
	plt.figure()
	plt.title("Recursive Feature Elimination Results")
	plt.xlabel("Features Selected")
	plt.ylabel("Negative Mean Squared Error")
	x_labels = [i for i in range(n_features, min_features_to_select, step * -1)]
	x_labels = [min_features_to_select] + x_labels[::-1]
	pdf = matplotlib.backends.backend_pdf.PdfPages("./charts/rfecv_results.pdf")
	plt.plot(x_labels, rfecv.grid_scores_)
	plt.tight_layout()
	pdf.savefig()
	pdf.close()
	plt.close("all")

	##PERFORM RECURSIVE FEATURE ELMINATION (RFE) 
	n_features_to_select = 10
	rfe = RFE(estimator = estimator, step =5, verbose = 1, n_features_to_select = n_features_to_select)
	rfe.fit(data_X, data_y)
	labels = data_X.columns[rfe.support_]
	print("Selected Columns: ")
	print(labels)
	data_X = data_X[labels]



	##############################################
	##### CREATE TEST AND TRAINING DATA ##########
	##############################################
	random_state = 9876
	train_X, test_X, train_y, test_y  = train_test_split(data_X, data_y, test_size=0.3, random_state=random_state)




	###############################################
	########## MODEL: RANDOM FOREST ###############
	###############################################
	if True:
		random_state = 876543
		estimator = RandomForestRegressor(random_state=random_state)
		param_grid = {"max_depth": [2,3],
						"n_estimators": [50,100,250,500],
						"max_features":["auto", "sqrt"],
						}
		grid_search = GridSearchCV(estimator, n_jobs=1, param_grid=param_grid, refit=True, verbose=2, cv=5, scoring='r2')
		grid_search.fit(train_X, train_y)
		print("Grid Search Results for Random Forest:")
		print("Best Parameters:")
		print(grid_search.best_params_)
		print("Training Score: %s " %(grid_search.best_score_))
		print("Training Time: %s secs" %(grid_search.cv_results_["mean_fit_time"][grid_search.best_index_]))
		evaluation_r2 = r2_score(test_y, grid_search.predict(test_X))
		print("Testing Score: %s" %(evaluation_r2) )
	
		##RESULTS:
		##BEST PARAMS = {'max_features': 'auto', 'max_depth': 3, 'n_estimators': 500}
		##TRAINING SCORE = 0.3261 R2
		##TRAINING TIME = 91.29 seconds
		##TESTING SCORE = 0.3275 R2
	
	
	###############################################
	########## MODEL: ADA BOOSTING ################
	###############################################
	if True:
		random_state = 1063
		estimator = AdaBoostRegressor(random_state=random_state)
		param_grid = {"n_estimators": [50,250,500],
						"learning_rate": [.1,1,10],
						"loss":["linear"],
						}
		grid_search = GridSearchCV(estimator, n_jobs=1, param_grid=param_grid, refit=True, verbose=2, cv=3, scoring='r2')
		grid_search.fit(train_X, train_y)
		print("Grid Search Results for ADA Boosting:")
		print("Best Parameters:")
		print(grid_search.best_params_)
		print("Training Score: %s " %(grid_search.best_score_))
		print("Training Time: %s secs" %(grid_search.cv_results_["mean_fit_time"][grid_search.best_index_]))
		evaluation_r2 = r2_score(test_y, grid_search.predict(test_X))
		print("Testing Score: %s" %(evaluation_r2) )
		
		##RESULTS:
		##BEST PARAMS = {'learning_rate': 0.1, 'loss': 'linear', 'n_estimators': 50}
		##TRAINING SCORE = 0.3573 R2
		##TRAINING TIME = 17.72 seconds
		##TESTING SCORE = 0.3573 R2

	
	
	###############################################
	########## MODEL: GRADIENT BOOSTING ###########
	###############################################
	if True:
		random_state = 1063
		estimator = GradientBoostingRegressor(random_state=random_state)
		param_grid = {"n_estimators": [100,300,500],
						"learning_rate": [.1,1],
						"loss":["ls"],
						"max_depth":[2,3]
						}
		grid_search = GridSearchCV(estimator, n_jobs=1, param_grid=param_grid, refit=True, verbose=2, cv=3, scoring='r2')
		grid_search.fit(train_X, train_y)
		print("Grid Search Results for Gradient Boosting:")
		print("Best Parameters:")
		print(grid_search.best_params_)
		print("Training Score: %s " %(grid_search.best_score_))
		print("Training Time: %s secs" %(grid_search.cv_results_["mean_fit_time"][grid_search.best_index_]))
		evaluation_r2 = r2_score(test_y, grid_search.predict(test_X))
		print("Testing Score: %s" %(evaluation_r2) )
		
		##RESULTS:
		##BEST PARAMS = {loss: ls, learning_rate: 0.1, n_estimators: 500, max_depth: 3}
		##TRAINING SCORE = 0.4868 R2
		##TRAINING TIME = 49.89 seconds
		##TESTING SCORE = 0.4882 R2

	
	###############################################
	######## MODEL: K NEAREST NEIGHBORS ###########
	###############################################
	if True:
		##PREPROCESS DATA
		cat_features = ["NumericListingCategory"]
		num_features = ["EmploymentStatusDuration", "CreditScoreRangeLower", "TotalCreditLinespast7years", "OpenRevolvingMonthlyPayment", "InquiriesLast6Months", "RevolvingCreditBalance", "AvailableBankcardCredit","DebtToIncomeRatio","StatedMonthlyIncome"]
		oneHotEncoding = make_column_transformer((OneHotEncoder(), cat_features),(StandardScaler(), num_features), sparse_threshold = 0)
		data_X_prep = oneHotEncoding.fit_transform(train_X) 
		data_X_prep = pd.DataFrame(data_X_prep)
		
		##PERFORM MODELING
		random_state = 1063
		estimator = KNeighborsRegressor()
		param_grid = {"n_neighbors": [5,10,25,50,100,200],
						"weights": ["uniform","distance"],
						}
		grid_search = GridSearchCV(estimator, n_jobs=1, param_grid=param_grid, refit=True, verbose=2, cv=3, scoring='r2')
		grid_search.fit(data_X_prep, train_y)
		print("Grid Search Results for Gradient Boosting:")
		print("Best Parameters:")
		print(grid_search.best_params_)
		print("Training Score: %s " %(grid_search.best_score_))
		print("Training Time: %s secs" %(grid_search.cv_results_["mean_fit_time"][grid_search.best_index_]))
		evaluation_r2 = r2_score(test_y, grid_search.predict(oneHotEncoding.transform(test_X)))
		print("Testing Score: %s" %(evaluation_r2) )

		##RESULTS:
		##BEST PARAMS = {weights: distance, n_neighbors: 50}
		##TRAINING SCORE = 0.4316 R2
		##TRAINING TIME = 0.48 seconds
		##TESTING SCORE = 0.4371 R2
		
		
if __name__ == "__main__":
	main()


