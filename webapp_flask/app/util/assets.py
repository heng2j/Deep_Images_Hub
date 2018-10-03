from flask_assets import Bundle, Environment
from .. import app

bundles = {

    'home_js': Bundle(
        'js/jquery/jquery-2.2.4.min.js',
        'js/active.js',
        'js/boostrap.min.js',
        'js/classy-nav.min.js',
        'js/map-active.js',
        'js/plugins.js',
        'js/popper.min.js',
        output='gen/home.js'),

    'home_css': Bundle(
        'css/lib/reset.css',
        'css/animate.css',
        'css/bootstrap.min.css',
        output='gen/home.css'),

    'admin_js': Bundle(
        'js/lib/jquery-1.10.2.js',
        'js/lib/Chart.js',
        'js/admin.js',
        output='gen/admin.js'),

    'admin_css': Bundle(
        'css/lib/reset.css',
        'css/common.css',
        'css/admin.css',
        output='gen/admin.css')
}

assets = Environment(app)

assets.register(bundles)