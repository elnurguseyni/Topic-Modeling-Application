from flask import Flask, request, jsonify
import subprocess
import tempfile
import json

app = Flask(__name__)

@app.route("/run_stm", methods=["POST"])
def run_stm():
    docs = request.json["documents"]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        for doc in docs:
            tmp.write(doc + "\n")
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["Rscript", "stm_runner.R", tmp_path],
            capture_output=True, text=True, check=True
        )
        return jsonify({"topics": json.loads(result.stdout)})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": e.stderr}), 500
    except json.JSONDecodeError:
        return jsonify({"error": f"Invalid JSON from R:\n{result.stdout}"}), 500

if __name__ == "__main__":
    print("✅ STM Flask server is running at http://localhost:5050")
    app.run(port=5050, debug=True)