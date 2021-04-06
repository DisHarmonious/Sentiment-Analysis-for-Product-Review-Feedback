from flask import Flask, Response, stream_with_context, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
import sqlite3, os

app=Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)

os.chdir(r"C:\Users\User\Desktop\pyspark_project\my_module")
con=sqlite3.connect("db.sqlite", check_same_thread=False)
cur = con.cursor()

class comments(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    comment=db.Column(db.String(1000))
    processed=db.Column(db.String(1))

@app.route("/", methods=["GET", "POST"])
def home_page():
    return render_template('main_page.html')

@app.route("/add", methods=["POST"])
def add():
    cur.execute("select MAX(id) from comments")
    maxid=cur.fetchall()
    id=maxid[0][0]+1
    comment=request.form.get('text')
    new_comment=comments(id=id, comment=comment, processed="0")
    db.session.add(new_comment)
    db.session.commit()
    return redirect(url_for("home_page"))

if __name__=="__main__":
    db.create_all()
    app.run(debug=True)