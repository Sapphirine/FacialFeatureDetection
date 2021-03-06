from flask import Flask, render_template, Response, url_for
from webcam import WebCam

# TODO: multi-threading, socket, frame rate
# TODO: start, stop, record, save
# TODO: more features (register, login)
# TODO: tolerance adjustment
# TODO: log visitors

app = Flask(__name__)
path = "C:\\Users\\Linsu Han\\PycharmProjects\\capture"
cascade_path = path + '\\resources\\haarcascade_frontalface_default.xml'
model = 'resnet50'
tolerance = 3
posts = [{'name': 'Linsu Han', 'uni': 'lh2910'},
         {'name': 'Zora Li', 'uni': 'xl2788'}]


@app.route('/')
def index():
    return render_template('index.html', posts=posts)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/detect/<feature>')
def detect(feature):
    Feature = ' '.join(feature.split('_')).title()
    return render_template('detect.html', feature=feature, Feature=Feature)


@app.route('/webcam/<feature>')
def webcam(feature):
    Feature = ' '.join(feature.split('_')).title()
    learner_path = path + f'\\resources\\models\\{feature}_{model}.pkl'
    cam = WebCam(Feature, learner_path, cascade_path, tolerance)
    return Response(cam.feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)  # set false in production
