from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Welcome! :). <br />More coming Soon.'


if __name__ == '__main__':
    app.run()
