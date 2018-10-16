from flask import render_template
from app import app
from app.forms import LoginForm
from app.DAO import get_featured_batches, get_all_labels, get_parent_labels , get_recent_trained_models , get_trained_models_by_id , get_trained_models_labels_sample_image_thumbnails_by_id


# TODO - Fix this qucik and dirty ways to make query with proper SQLAlchemy style or should use Node JS with PostRest

@app.route('/')
@app.route('/index.html')
def index():
    user = {'username': 'Heng'}


    features = get_featured_batches()


    return render_template('index.html', title='Home', user=user, features=features)

@app.route('/')
def static_file():
    return app.send_static_file('index.html')


@app.route('/')
@app.route('/shop.html')
def labels():

    all_labels = get_all_labels()

    parent_labels = get_parent_labels()

    return render_template('shop.html', title='Labels Collection', all_labels=all_labels, parent_labels = parent_labels)


@app.route('/')
@app.route('/cart.html')
def trained_models():

    model_list = get_recent_trained_models()


    return render_template('cart.html', title='Trained Models', model_list=model_list)

@app.route('/')
@app.route('/product-details.html/<model_id>')
def model_details(model_id):

    model_details = get_trained_models_by_id(model_id)
    image_thumbnails = get_trained_models_labels_sample_image_thumbnails_by_id(model_id)



    return render_template('product-details.html', title='Trained Models', model_details=model_details , image_thumbnails=image_thumbnails)




if __name__ == '__main__':
   app.run(debug = True)