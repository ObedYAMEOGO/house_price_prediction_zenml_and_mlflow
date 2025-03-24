from flask import Flask, render_template, request, jsonify
import json
import requests

app = Flask(__name__)

# MLflow prediction server URL
MLFLOW_URL = "http://127.0.0.1:8000/invocations"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Convert form data to the expected format
        input_data = {
            "dataframe_records": [{
                "Order": int(form_data.get('Order', 1)),
                "PID": int(form_data.get('PID', 5286)),
                "MS SubClass": int(form_data.get('MS_SubClass', 20)),
                "Lot Frontage": float(form_data.get('Lot_Frontage', 80.0)),
                "Lot Area": int(form_data.get('Lot_Area', 9600)),
                "Overall Qual": int(form_data.get('Overall_Qual', 5)),
                "Overall Cond": int(form_data.get('Overall_Cond', 7)),
                "Year Built": int(form_data.get('Year_Built', 1961)),
                "Year Remod/Add": int(form_data.get('Year_Remod_Add', 1961)),
                "Mas Vnr Area": float(form_data.get('Mas_Vnr_Area', 0.0)),
                "BsmtFin SF 1": float(form_data.get('BsmtFin_SF_1', 700.0)),
                "BsmtFin SF 2": float(form_data.get('BsmtFin_SF_2', 0.0)),
                "Bsmt Unf SF": float(form_data.get('Bsmt_Unf_SF', 150.0)),
                "Total Bsmt SF": float(form_data.get('Total_Bsmt_SF', 850.0)),
                "1st Flr SF": int(form_data.get('1st_Flr_SF', 856)),
                "2nd Flr SF": int(form_data.get('2nd_Flr_SF', 854)),
                "Low Qual Fin SF": int(form_data.get('Low_Qual_Fin_SF', 0)),
                "Gr Liv Area": float(form_data.get('Gr_Liv_Area', 1710.0)),
                "Bsmt Full Bath": int(form_data.get('Bsmt_Full_Bath', 1)),
                "Bsmt Half Bath": int(form_data.get('Bsmt_Half_Bath', 0)),
                "Full Bath": int(form_data.get('Full_Bath', 1)),
                "Half Bath": int(form_data.get('Half_Bath', 0)),
                "Bedroom AbvGr": int(form_data.get('Bedroom_AbvGr', 3)),
                "Kitchen AbvGr": int(form_data.get('Kitchen_AbvGr', 1)),
                "TotRms AbvGrd": int(form_data.get('TotRms_AbvGrd', 7)),
                "Fireplaces": int(form_data.get('Fireplaces', 2)),
                "Garage Yr Blt": int(form_data.get('Garage_Yr_Blt', 1961)),
                "Garage Cars": int(form_data.get('Garage_Cars', 2)),
                "Garage Area": float(form_data.get('Garage_Area', 500.0)),
                "Wood Deck SF": float(form_data.get('Wood_Deck_SF', 210.0)),
                "Open Porch SF": int(form_data.get('Open_Porch_SF', 0)),
                "Enclosed Porch": int(form_data.get('Enclosed_Porch', 0)),
                "3Ssn Porch": int(form_data.get('3Ssn_Porch', 0)),
                "Screen Porch": int(form_data.get('Screen_Porch', 0)),
                "Pool Area": int(form_data.get('Pool_Area', 0)),
                "Misc Val": int(form_data.get('Misc_Val', 0)),
                "Mo Sold": int(form_data.get('Mo_Sold', 5)),
                "Yr Sold": int(form_data.get('Yr_Sold', 2010)),
            }]
        }

        # Convert to JSON
        json_data = json.dumps(input_data)
        headers = {"Content-Type": "application/json"}

        # Send to MLflow server
        response = requests.post(MLFLOW_URL, headers=headers, data=json_data)

        if response.status_code == 200:
            prediction = response.json()
            return jsonify({
                'success': True,
                'prediction': prediction,
                'input_data': input_data
            })
        else:
            return jsonify({
                'success': False,
                'error': f"MLflow server error: {response.status_code}",
                'details': response.text
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)