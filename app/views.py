import os
from app import app
from flask import render_template, request, redirect, make_response, jsonify
from werkzeug.utils import secure_filename
from app import imgprocess

ALLOWED_IMAGE_EXTENSIONS = ["PNG", "JPG", "JPEG", "GIF"]


@app.route("/")
def index():
    return render_template('index.html')


def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]
    if ext.upper() in ALLOWED_IMAGE_EXTENSIONS:
        return True
    else:
        return False


@app.route("/compare", methods=["POST"])
def compare():

    if request.files:
        left_img = request.files['left_img']
        right_img = request.files['right_img']

        if left_img.filename == "" or right_img.filename == "":
            print("Image must have a filename")
            return redirect('/')

        if not allowed_image(left_img.filename) or not allowed_image(right_img.filename):
            print("Please upload allowed file types")
            return redirect('/')
        else:
            upload_dir = os.path.join(os.getcwd(), 'app/static/uploads')
            left_img_name = secure_filename(left_img.filename)
            left_img_name = os.path.join(upload_dir, left_img_name)
            left_img.save(left_img_name)
            right_img_name = secure_filename(right_img.filename)
            right_img_name = os.path.join(upload_dir, right_img_name)
            right_img.save(right_img_name)

            # Image comparison using SSIM
            if request.form['algorithm'] == 'alg_ssim':
                diff_areas = imgprocess.image_compare_ssim(
                    left_img_name, right_img_name)
                print(jsonify(diff_areas))
                response = make_response(jsonify(diff_areas), 200)
                return response
            # Image comparison using AbsDiff
            elif request.form['algorithm'] == 'alg_absdiff':
                diff_areas = imgprocess.image_compare_absdiff(
                    left_img_name, right_img_name)
                print(jsonify(diff_areas))
                response = make_response(jsonify(diff_areas), 200)
                return response
            # Image compoarison algorithm is not specified
            else:
                print("Please select an image comparison algorithm")
                return redirect('/')

    return redirect('/')
