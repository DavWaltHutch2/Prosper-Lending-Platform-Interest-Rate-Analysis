import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf

def main():
	#########################################
	########## SETUP ENVIRONMENT ############
	#########################################

	##SET SEED
	random_state = 876539

	##GET DATA
	data = pd.read_csv('./data/ProsperLoanData.csv')


	##RENAME FIELDS
	data.rename(columns = {"ProsperRating (numeric)": "NumericProsperRating", "ProsperRating (Alpha)": "AlpaProsperRating", "ListingCategory (numeric)": "NumricListingCategory"}, inplace = True)


	##GET RELEVANT FIELDS
	data = data[["Term","BorrowerRate","NumericProsperRating","AlpaProsperRating","ProsperScore","NumricListingCategory","BorrowerState","Occupation",
				"EmploymentStatus","EmploymentStatusDuration","IsBorrowerHomeowner","CreditScoreRangeLower","CurrentCreditLines","OpenCreditLines",
				"TotalCreditLinespast7years","OpenRevolvingAccounts","OpenRevolvingMonthlyPayment","InquiriesLast6Months","TotalInquiries","CurrentDelinquencies","AmountDelinquent",
				"DelinquenciesLast7Years","PublicRecordsLast10Years","PublicRecordsLast12Months","RevolvingCreditBalance","BankcardUtilization","AvailableBankcardCredit",
				"DebtToIncomeRatio","IncomeRange","StatedMonthlyIncome"]]



	##########################################
	############  EXPLORATION ################
	##########################################

	##CORROLATION PLOT
	data_num = data[["BorrowerRate","EmploymentStatusDuration","CreditScoreRangeLower","CurrentCreditLines",
											 "OpenCreditLines","TotalCreditLinespast7years","OpenRevolvingAccounts","OpenRevolvingMonthlyPayment",
											 "InquiriesLast6Months","TotalInquiries","CurrentDelinquencies","AmountDelinquent","DelinquenciesLast7Years",
											 "PublicRecordsLast10Years","PublicRecordsLast12Months","RevolvingCreditBalance","BankcardUtilization",
											 "AvailableBankcardCredit","DebtToIncomeRatio","StatedMonthlyIncome"]]
											 

	fig, ax = plt.subplots()
	fig.set_size_inches(15, 15)
	ax = sns.heatmap(data_num.corr(),vmin = -1, vmax = 1, center = 0, cmap=sns.diverging_palette(15, 150, center = "light", as_cmap=True))
	ax.set_title("Prosper Correlation Plot", fontsize = 50)
	fig.tight_layout()
	fig.savefig("./charts/prosper_correlation_plot.png")
	plt.close("all")

	
	##BOXPLOTS: BORROWER RATE VS TERM 
	data.boxplot(column = ["BorrowerRate"],by = "Term"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 5, rotation="vertical", horizontalalignment='center')
	ax.set_ylabel("Borrower Rate")
	ax.set_title("Term vs Borrower Rate")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS PROSPERSCORE 
	data.boxplot(column = ["BorrowerRate"],by = "ProsperScore"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 5, rotation="vertical", horizontalalignment='center')
	ax.set_ylabel("Borrower Rate")
	ax.set_title("Prosper Score vs Borrower Rate")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS BORROWER STATE 
	data.boxplot(column = ["BorrowerRate"],by = "BorrowerState"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 5, rotation="vertical", horizontalalignment='center')
	ax.set_ylabel("Borrower State")
	ax.set_title("Prosper Score vs Borrower State")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS OCCUPATION 
	data.boxplot(column = ["BorrowerRate"],by = "Occupation"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 5, rotation="vertical", horizontalalignment='center')
	ax.set_ylabel("Occupation")
	ax.set_title("Prosper Score vs Occupation")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS EMPLOYMENT STATUS 
	data.boxplot(column = ["BorrowerRate"],by = "EmploymentStatus"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 5, rotation="vertical", horizontalalignment='center')
	ax.set_ylabel("Employmnet Stauts")
	ax.set_title("Prosper Score vs Employment Status")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	
	##BOXPLOTS: BORROWER RATE VS HOMEOWNER TYPE
	data.boxplot(column = ["BorrowerRate"],by = "IsBorrowerHomeowner"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("IsBorrowerHomeowner")
	ax.set_title("Prosper Score vs Homeowner Type")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS CREDIT SCORE
	data.boxplot(column = ["BorrowerRate"],by = "CreditScoreRangeLower"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("CreditScoreRangeLower")
	ax.set_title("Prosper Score vs Credit Score")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS INCOME RANGE
	data.boxplot(column = ["BorrowerRate"],by = "IncomeRange"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("IncomeRange")
	ax.set_title("Prosper Score vs Income")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS NUMERIC PROSPER RATING
	data.boxplot(column = ["BorrowerRate"],by = "NumericProsperRating"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("Prosper Rating (Numeric)")
	ax.set_title("Prosper Score vs Prosper Rating (Numeric)")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS ALPHA PROSPER RATING
	data.boxplot(column = ["BorrowerRate"],by = "AlpaProsperRating"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("Prosper Rating (Alpha)")
	ax.set_title("Prosper Score vs Prosper Rating (Alpha)")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	##BOXPLOTS: BORROWER RATE VS NUMERIC LISTING CATEGORY
	data.boxplot(column = ["BorrowerRate"],by = "NumricListingCategory"); 
	ax = plt.gca()
	plt.setp(ax.get_xticklabels(), fontsize = 10, rotation= 45, horizontalalignment='center')
	ax.set_ylabel("Numeric Listing Category")
	ax.set_title("Prosper Score vs Listing Category")
	fig = plt.gcf()
	fig.suptitle("")
	fig.tight_layout()
	
	pdf = matplotlib.backends.backend_pdf.PdfPages("./charts/box_plots.pdf")
	for fig_id in plt.get_fignums():
		pdf.savefig( fig_id )
	pdf.close()
	plt.close("all")





if __name__ == "__main__":
	main()


