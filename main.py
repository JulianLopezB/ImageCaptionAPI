
import io
import torch
from flask import Flask, request, jsonify, render_template, redirect
from werkzeug.utils import secure_filename
from feature_extractor import FeatureExtractor
import captioning
import captioning.utils.misc
import captioning.models
from src.utils import *
import sys
import os
sys.path.append('vqa-maskrcnn-benchmark')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_extractor = FeatureExtractor(device=device)

infos = captioning.utils.misc.pickle_load(open('model_data/infos_trans12-best.pkl', 'rb'))
infos['opt'].vocab = infos['vocab']

model = captioning.models.setup(infos['opt'])
#model.cuda()
model.to(device)
model.load_state_dict(torch.load('model_data/model-best.pth', map_location=torch.device(device)))


app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')

def refresh_paths():
    os.makedirs('static/tmp', exist_ok=True)
    clean_path_content('static/tmp')
    os.makedirs('static/tmp/frames', exist_ok=True)
    print(f'Uploading temporary files to {uploads_dir}')
    os.makedirs(uploads_dir, exist_ok=True)

def get_img_predictions(img_url):
    try:
        #image_path = feature_extractor.get_actual_image(img_url)
        prediction = '<br>'.join(get_captions(feature_extractor(img_url)))
        data = {"prediction": prediction}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

def get_captions(img_feature):
    # Return the 5 captions from beam serach with beam size 5
    return model.decode_sequence(model(img_feature.mean(0)[None], img_feature[None], mode='sample', opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':5})[0])



@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
        refresh_paths()
        filename = 'tmp.png'
        img_input = os.path.join(uploads_dir, secure_filename(filename))
        file.save(img_input)
        predictions = get_img_predictions(img_input)
        return predictions
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    #app.run(debug=True)
    