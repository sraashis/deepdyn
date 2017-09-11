from flask import Flask

import app_config as conf

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/')
def hello_world():
    return 'Welcome! :). <br />More coming Soon'


if __name__ == '__main__':
    app.run()
