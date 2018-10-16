from flask import render_template
from app import app
from app.forms import LoginForm
from app.DAO import get_featured_batches, get_all_labels, get_parent_labels , get_recent_trained_models


# TODO - Fix his qucik and dirty ways to make query with proper SQLAlchemy style

@app.route('/')
@app.route('/index.html')
def index():
    user = {'username': 'Heng'}


    features = get_featured_batches()


    return render_template('index.html', title='Home', user=user, features=features)

@app.route('/')
def static_file():
    return app.send_static_file('index.html')

@app.route('/login')
def login():
    form = LoginForm()
    return render_template('login.html', title='Sign In', form=form)

@app.route('/')
@app.route('/shop.html')
def shop():

    all_labels = get_all_labels()

    parent_labels = get_parent_labels()

    return render_template('shop.html', title='Shop', all_labels=all_labels, parent_labels = parent_labels)


@app.route('/')
@app.route('/cart.html')
def cart():

    model_list = get_recent_trained_models()

    return render_template('cart.html', title='Trained Models', model_list=model_list)



if __name__ == '__main__':
   app.run(debug = True)