from flask import Flask, render_template, request
import numpy as np
import pickle
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/',  methods = ['POST', 'GET'])
@app.route('/index',  methods = ['POST', 'GET'])
@cross_origin()
def index():

	if request.method == "POST":
		
		##GET MODEL
		model = None
		with open('./model/model.pckl','rb') as f:
			model = pickle.load(f)
	
		##CALCULATE VALUES
		LoanOriginalAmount = request.form["loanAmt"]
		EmploymentStatusDuration = request.form["employmentLength"] * 12
		StatedMonthlyIncome = request.form["income"]
		CreditScoreRangeLower = request.form["creditScore"]
		InquiresLast6Months = request.form["loans6Months"]
		RevolvingCreditBalance = request.form["sumBalanceCards"]
		AvailableBankcardCredit = request.form["sumLimitsCards"]
		DebtToIncomeRatio = float(request.form["sumBills"]) / float(request.form["income"])
		TotalCreditLinespast7years = request.form["credit7Years"]
		OpenRevolvingMonthlyPayment = request.form["sumBills"]
		
		##MAKE PREDICTIONS
		##EmploymentStatusDuration, CreditScoreRangeLower, TotalCreditLinespast7years, OpenRevolvingMonthlyPayment, InquiresLast6Months,
		##RevolvingCreditBalance, AvailableBankcardCredit, DebtToIncomeRatio, StatedMonthlyIncome, LoanOriginalAmount
		input = np.array([EmploymentStatusDuration, CreditScoreRangeLower, TotalCreditLinespast7years, OpenRevolvingMonthlyPayment, InquiresLast6Months,
					RevolvingCreditBalance, AvailableBankcardCredit, DebtToIncomeRatio, StatedMonthlyIncome, LoanOriginalAmount]).reshape(1, -1)
		
		prediction = model.predict(input)
		return render_template('index.html', prediction = round(prediction[0] * 100, 2))
	else: 
		return render_template("index.html")




if __name__ == "__main__":
    app.run(debug=True)
