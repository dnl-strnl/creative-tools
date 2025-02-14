from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    svg_content = request.form['svg']
    return jsonify(dict(svg=svg_content))

if __name__ == '__main__':
    app.run(debug=True)
