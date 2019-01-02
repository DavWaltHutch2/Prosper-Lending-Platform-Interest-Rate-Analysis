import pickle
import pandas as pd
import matplotlib; matplotlib.use('Agg')
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from collections import defaultdict


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
column_list = ["NumericProsperRating","AlpaProsperRating","ProsperScore", "NumericListingCategory"]
print("Columns >>")
print(data.columns)
for col in column_list:
	if any(data.columns.isin([col])):
		data = data.drop([col], axis = 1)

	
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
multilabel_encoder = MultiLabelEncoder(cat_features)
data = multilabel_encoder.fit_transform(data)
		
		
##SEPARATE X AND Y
data_y = data["BorrowerRate"]
data_X = data.drop(["BorrowerRate"], axis = 1)




###############################################
#### PERFORM RECURSIVE FEATURE ELIMINATION ####
###############################################

##CREATE VARIABLES
random_state = 2865325
estimator = DecisionTreeRegressor(random_state = random_state)


##PERFORM RECURSIVE FEATURE ELMINATION (RFE) 
n_features_to_select = 10
rfe = RFE(estimator = estimator, step =3, verbose = 1, n_features_to_select = n_features_to_select)
rfe.fit(data_X, data_y)
labels = data_X.columns[rfe.support_]
print("Selected Columns: ")
print(labels)
data_X = data_X[labels]


###############################################
################# MODEL #######################
###############################################
##MODEL: Gradient Boosting 
##BEST PARAMS = {loss: ls, learning_rate: 0.1, n_estimators: 500, max_depth: 3}
##TRAINING SCORE = 0.4868 R2
##TRAINING TIME = 49.89 seconds
##TESTING SCORE = 0.4882 R2

random_state = 1063
model = GradientBoostingRegressor(random_state=random_state, n_estimators =500, max_depth = 3, learning_rate = 0.1, loss = "ls")
model.fit(data_X, data_y)
filename = "./model/model.pckl"
pickle.dump(model, open(filename, 'wb'))





