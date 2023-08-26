from pulp import *
from fractions import Fraction
import pickle


def get_price_market_A(month):
    # load model
   # market_A_model = ARIMAResults.load('market_A_model.pkl')
    with open('market_A_model.pkl', 'rb') as file:
        data = pickle.load(file)
    
    fc_series = data['forecast_series']
    pred123 = fc_series.get(key = month)
    return pred123

def get_price_market_B(month):
    # load model
   # market_A_model = ARIMAResults.load('market_A_model.pkl')
    with open('market_B_model.pkl', 'rb') as file:
        data = pickle.load(file)
    
    fc_series = data['forecast_series']
    pred123 = fc_series.get(key = month)
    return pred123
    
def get_price_market_C(month):
    # load model
   # market_A_model = ARIMAResults.load('market_A_model.pkl')
    with open('market_C_model.pkl', 'rb') as file:
        data = pickle.load(file)
    
    fc_series = data['forecast_series']
    pred123 = fc_series.get(key = month)
    return pred123

cost = float(input("Enter Cost: "))
Month=input("Enter Month: ")  
x = int(input("Bigger + Green Pineapples: "))
z = int(input("Small + Green pineapples: "))

print(" ")

print("Predicting the Prices for each Pineapple Market (According to the harvested month)")
#p1 = getPrice_arima(Month) #fresh_export_price

p1 = get_price_market_A(Month) 
p2 = get_price_market_B(Month) 
p3 = get_price_market_C(Month) 

#Optimization Process
print("Optimization Process Begin...")
print("")
profit = LpProblem("Example of standard maximum problem",LpMaximize)
print("Initial Profit : ", profit)

# nonnegativity constraints

z1=LpVariable("z1",1)
z2=LpVariable("z2",1)

# objective function
profit += ((p1*x + p2*z1 + p3*z2) - cost)

zero = 0
# main constraints

profit += z1+z2 <= z, "constraint 1"

#double check the problem
print (profit)

# The problem is solved using PuLP's choice of Solver
profit.solve()

# status of the solution
print(f"Status: {LpStatus[profit.status]}")

#Individual decision_variables
print ("Individual decision_variables: ")
for v in profit.variables():
    print(v.name, "=", v.varValue)
    
# maximum value of the objective function
print("Optimal Profit for the problem: ", value(profit.objective))


print(" ")
print("******************************************************************************")
total_amount = x + z
fresh_market_income = x * p1
process_market_income_1 = profit.variables()[0].varValue * p2
process_market_income_2 = profit.variables()[1].varValue * p3
total_income = fresh_market_income + process_market_income_1 + process_market_income_2 
print("***** Optimal Product Differentiation Plan *****")
print(" ")
print("Total quantity of Pineapple : " , total_amount)
print("Quantity of Pineapples for Fresh Pineapple Market        : " , x)
print("Quantity of Pineapples for Processed Pineapple Market A  : " , profit.variables()[0].varValue)
print("Quantity of Pineapples for Processed Pineapple Market B  : " , profit.variables()[1].varValue)
print(" ")
print("Predicted Prices Per 1 kg")
print("Fresh Market Price (According to the harvested season)      : Rs.", round(p1,2))
print("Process Market A Price  (According to the harvested season) : Rs.", round(p2,2))
print("Process Market B Price  (According to the harvested season) : Rs.", round(p3,2))
print(" ")
print("Fresh Market Income                                        : Rs.", round(fresh_market_income,2))
print("Process Market Income  A                                   : Rs.", round(process_market_income_1,2))
print("Process Market Income  b                                   : Rs.", round(process_market_income_2,2))
print(" ")
print("Total Income   : Rs.", round(total_income,2))
print("Total Profit   : Rs.", round(value(profit.objective),2))
print("**********************************************************************************")
print("")
