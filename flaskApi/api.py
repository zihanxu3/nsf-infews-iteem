import time
from flask import Flask, request
from ITEEM import ITEEM
import numpy as np
from flask import jsonify
import json


app = Flask(__name__)

application = app

baseline_global = None

@app.route("/")
def hello():
    return "Congratulation! You service Hosted on CPanel"

@app.route('/simulate', methods=['POST'])
def simulate():
    print("function gets called at least")
    

    matrix = request.get_json()['subwatershed']
    tech_wwt = request.get_json()['wwt_param']
    tech_GP1 = request.get_json()['nwwt_param_wmp1']
    tech_GP2 = request.get_json()['nwwt_param_wmp2']
    tech_GP3 = request.get_json()['nwwt_param_dmp']

    landuse_matrix_baseline = np.zeros((45,62))
    temp_matrix = np.array(matrix) / 100
    landuse_matrix_baseline[:, 0] = temp_matrix[:, 0]
    landuse_matrix_baseline[:, 37] = temp_matrix[:, 1]
    landuse_matrix_baseline[:, 39] = temp_matrix[:, 2]
    landuse_matrix_baseline[:, 46] = temp_matrix[:, 3]
    landuse_matrix_baseline[:, 47] = temp_matrix[:, 4]
    landuse_matrix_baseline[:, 48] = temp_matrix[:, 5]
    landuse_matrix_baseline[:, 55] = temp_matrix[:, 6]



    baseline = ITEEM(landuse_matrix_baseline, tech_wwt=tech_wwt, limit_N=10.0, tech_GP1=tech_GP1, tech_GP2=tech_GP2, tech_GP3=tech_GP3)
    baseline_global = baseline

    yield_data = {}
    yield_data['nitrate'] = baseline.get_yield('nitrate')
    yield_data['phosphorus'] = baseline.get_yield('phosphorus')
    yield_data['streamflow'] = baseline.get_yield('streamflow')

    sediment_load = baseline.get_sediment_load()

    load_data = {}
    load_data['nitrate'] = baseline.get_load('nitrate')
    load_data['phosphorus'] = baseline.get_load('phosphorus')
    load_data['streamflow'] = baseline.get_load('streamflow')

    load_data_list = []
    for i in range(45):
        load_data_list.append(
            {
                "nitrate": {
                    "name": "P Flow Nitrate",
                    "data": np.array(load_data['nitrate'])[:, :, i].flatten().tolist(), 
                },
                "phosphorus": {
                    "name": "P Flow Phosphorus",
                    "data": np.array(load_data['phosphorus'])[:, :, i].flatten().tolist(), 
                },
                "streamflow": {
                    "name": "P Flow Streamflow",
                    "data": np.array(load_data['streamflow'])[:, :, i].flatten().tolist(), 
                },
            }
        )
    crop_yield = {}
    crop_production = {}
    crop_yield['corn'], crop_production['corn'] = baseline.get_crop('corn')
    crop_yield['soybean'], crop_production['soybean'] = baseline.get_crop('soybean')
    crop_yield['switchgrass'], crop_production['switchgrass'] = baseline.get_crop('switchgrass')

    # {
    #     "nitrate": {
    #         "name": "Nitrate P Yield Data",
    #         "data": yield_data['nitrate'], 
    #     },
    #     "phosphorus": {
    #         "name": "Phosphorus P Yield Data",
    #         "data": yield_data['phosphorus'], 
    #     },
    #     "streamflow": {
    #         "name": "Streamflow P Yield Data",
    #         "data": yield_data['streamflow'], 
    #     },
    # },
    P_list = []

    _, _, _, _, _, _, _, P_list = baseline.get_P_flow()
    P_list = [{
        "type": 'sankey',
        "keys": ['from', 'to', 'weight'],
        "data": P_list,
    }]
    output = {
        'yieldData': yield_data,
        'sedimentLoad': sediment_load, 
        'loadData': load_data, 
        'loadDataList': load_data_list,
        'cropYield': crop_yield, 
        'cropProduction': crop_production,
        'plist': P_list,
    }

    # with open('data.json', 'w') as outfile:
    #     json.dump(output, outfile)

    print(jsonify(output))
    return jsonify(output), 200

# import time
# from flask import Flask

# app = Flask(__name__)

# @app.route('/time')
# def get_current_time():
#     return {'time': time.time()}