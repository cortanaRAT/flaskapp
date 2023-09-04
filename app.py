from flask import Flask, render_template, request,redirect,url_for
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image
from mody.model.model import HumanSegment, HumanMatting
import mody.inference as infer
import numpy as np
from PIL import Image
import os
import utils
app = Flask(__name__)
bg=Image.open(r'static/bg.png')


def removebg(file,backbone='resnet50'):
    global bg
   
    model = HumanMatting(backbone=backbone)
    model = nn.DataParallel(model).cpu().eval()
    model.load_state_dict(torch.load(r"assests\SGHM-ResNet50.pth", map_location=torch.device('cpu')))
    print("Load checkpoint successfully ...")
    img = Image.open(file).convert("RGB")
    pred_alpha, pred_mask = infer.single_inference(model, img)
    alpha_image = Image.fromarray((pred_alpha * 255).astype('uint8'), mode='L')
    bg = bg.resize((img.size))
    im = Image.composite(img, bg, alpha_image)
    return im



# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"
@app.route('/loading')
def	loading():
	return render_template("index.html",pro=1)
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		imged=removebg(img_path)
		removedpath=os.path.join('static/removed',img.filename)
		rgb_image = imged.convert("RGB")
		rgb_image.save(removedpath)
	return render_template("index.html",img_path = removedpath)



if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)