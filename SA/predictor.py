import tensorflow as tf
import os, time, sqlite3, random

os.chdir(r"C:\Users\User\Desktop\SA_project")
#load models
check=1
if check==1:
    m1=tf.keras.models.load_model('models\CNN_SA_50e')
    m2=tf.keras.models.load_model('models\CNN_SA_20e')
check=0
print("hi")

#connect to sqlite
os.chdir(r"my_module")
conn = sqlite3.connect("db.sqlite")
cur = conn.cursor()

#create list to store predictions
predictions=[]

if __name__=="__main__":
    while 1==1:
        #get comments to process
        cur.execute("select * from comments where processed=0")
        rows = cur.fetchall()
        if len(rows)>0:
            ids=[row[0] for row in rows]
            for row in rows:
                #choose model
                x=random.choices([1, 2], weights=[.6, .4])
                #predict
                if x==1:
                    prediction=m1.predict([[row[1]]])
                    predictions.append([row[0], prediction])
                    print(predictions[-1])
                else:
                    prediction=m2.predict([[row[1]]])
                    predictions.append([row[0], prediction])
                    print(predictions[-1])
            #update table, marking out the processed comments
            for idn in ids:
                sql="""update comments set processed="1" where id= ?"""
                data=(str(idn),)
                cur.execute(sql,data)
                conn.commit()
        print("Waiting for new comments")
        time.sleep(5)

