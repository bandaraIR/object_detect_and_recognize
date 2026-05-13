# Object Detection & License Plate Recognition

## ⚠️ Note: iot_server.py

`iot_server.py` file is not included in this repository due to a Git push conflict error.

### What it does:
A simple Flask API server that controls a red light state for IoT devices.

**GET /red** — returns current red light state  
**POST /red** — updates red light state (`{"red_on": true}` or `{"status": "ON"}`)

### Code:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

RED_STATE = {"red_on": False}

@app.get("/red")
def get_red():
    return jsonify(RED_STATE)

@app.post("/red")
def set_red():
    data = request.get_json(silent=True) or {}

    if "red_on" in data:
        RED_STATE["red_on"] = bool(data["red_on"])
    elif "status" in data:
        RED_STATE["red_on"] = str(data["status"]).upper() == "ON"
    else:
        RED_STATE["red_on"] = False

    return jsonify({"ok": True, "red_on": RED_STATE["red_on"]})
```
