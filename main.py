
import os
import torch
from flask import Flask, request, jsonify
import captioning
import captioning.utils.misc
import captioning.models
import sys
sys.path.append('vqa-maskrcnn-benchmark')
from feature_extractor import FeatureExtractor


device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_extractor = FeatureExtractor(device=device)

infos = captioning.utils.misc.pickle_load(open('model_data/infos_trans12-best.pkl', 'rb'))
infos['opt'].vocab = infos['vocab']

model = captioning.models.setup(infos['opt'])
#model.cuda()
model.to(device)
model.load_state_dict(torch.load('model_data/model-best.pth', map_location=torch.device(device)))

def get_captions(img_feature):
    # Return the 5 captions from beam serach with beam size 5
    return model.decode_sequence(model(img_feature.mean(0)[None], img_feature[None], mode='sample', opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':5})[0])


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        img_url = request.args.get('img_url')
        if not img_url:
            return jsonify({"error": "no img url provided"})

        try:
            #image_path = feature_extractor.get_actual_image(img_url)
            prediction = '<br>'.join(get_captions(feature_extractor(img_url)))
            data = {"prediction": prediction}
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    #app.run(debug=True)
    