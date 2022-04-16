from flask import Flask, render_template, request
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

try:
    import shutil
    shutil.rmtree('public / static / img / upload')
    % cd public / static / img % mkdir upload % cd .. / ..
    print()
except:
    pass

model = tf.keras.models.load_model('model')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'public / static / img / upload'
ALLOWED_EXTENSIONS = {'jpg'}


@app.route('/')
def upload_f():
    return render_template('detector.html')


def finds():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # change this according to what you've trained your model to do
    vals = ['Positive', 'Negative']
    test_dir = 'public/'
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(-1, 256, 256, 3),
        shuffle=False,
        class_mode='categorical',
        batch_size=1)

    pred = model.predict_generator(test_generator)
    print(pred)
    return str(vals[np.argmax(pred)])


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds()
        return render_template('detector.html', ss=val)


if __name__ == '__main__':
    app.run()
