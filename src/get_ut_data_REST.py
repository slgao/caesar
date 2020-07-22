import requests
system = "Furis"
run_id = 42330
url_app = f"http://10.17.96.62:8181/application.wadl"
url_channels = f"http://10.17.96.62:8181/v1/channels/{system}"
url_systems = "http://10.17.96.62:8181/v1/systems"
url_runids = "http://10.17.96.62:8181/v1/runids"
url_data = "http://10.17.96.62:8181/v1/data"
url_columns = "http://10.17.96.62:8181/v1/columns"
url_meas_bname = f"http://10.17.96.62:8181/v1/measurementbasename/{run_id}"
url_meas_src = f"http://10.17.96.62:8181/v1/measurementsource/{run_id}"
pay_load = {"channel": "Object", "system": "Furis"}
echo_furis = {"channel": "Echo data", "system": "Furis"}
test_data = {
    "channel": "Geometry gauge",
    "key": "G1LFOGMS",
    "keyColumn": "Name",
    "runId": 518223,
    "system": "Plasser",
    "valueColumn": "Gauge"
}
Bscan_data = {
    "channel":"Object",
    "runId":42330,
    "system":"Furis",
    "valueColumn": "RunID"
}

test_compound_data = {
    "channel": "Geometry",
    "runId": 518223,
    "system": "Plasser",
    "valueColumn": "Curvature"
}
r_channels = requests.get(url_channels)
r_systems = requests.get(url_systems)
r_meas_src = requests.get(url_meas_src)
r_meas_bname = requests.get(url_meas_bname)
r_runids = requests.post(url_runids, json=pay_load)
# r_data = requests.post(url_data, json=test_data)
B_data = requests.post(url_data, json=Bscan_data)
r_columns = requests.post(url_columns, json=pay_load)
test_r_data = requests.post(url_data, json=test_compound_data)

