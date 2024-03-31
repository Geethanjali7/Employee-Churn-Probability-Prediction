import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

try:
    with open('empchurn.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        scaler = model_data['scaler']
except (IOError, EOFError) as e:
    print("Error Loading the pickled file!!!!", e)

@application.route('/')
def fun():
    return render_template('index.html')

@application.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        data = request.form
        features = [[int(data["Age"]), int(data["DistanceFromHome"]),
                     int(data["EnvironmentSatisfaction"]), int(data["JobInvolvement"]), int(data["JobLevel"]),
                     int(data["JobRole"]), int(data["JobSatisfaction"]), int(data["MonthlyIncome"]),
                     int(data["NumCompaniesWorked"]), int(data["OverTime"]), int(data["PercentSalaryHike"]),
                     int(data["PerformanceRating"]), int(data["RelationshipSatisfaction"]),
                     int(data["StockOptionLevel"]), int(data["TotalWorkingYears"]), int(data["TrainingTimesLastYear"]),
                     int(data["WorkLifeBalance"]), int(data["YearsAtCompany"]), int(data["YearsInCurrentRole"]),
                     int(data["YearsSinceLastPromotion"]), int(data["YearsWithCurrManager"])]]

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        prediction_percentage = prediction * 100
        return render_template("index.html", prediction=prediction_percentage)
    else:
        return "Method Not Allowed", 405

if __name__ == "__main__":
    application.run(debug=True)
