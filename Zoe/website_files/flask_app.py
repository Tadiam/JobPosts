from flask import Flask, request, session
from processing import process


app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "93e78914d7b162b36641ca4ec562c86d4f14a399e2629ca218742fbc0610168e"

#inputs = [] global variables bad!

@app.route("/", methods = ["GET", "POST"])
#what follows is what happens when someone goes to the / on your site
def hello_world():
    if "inputs" not in session:
        session["inputs"] = []
        session["position_name"] = ""
        session["job_category"] = ""
        session["salary"] = ""
        session["type_job"] = ""
        session["shift_schedule"] = ""
        session["location"] = ""
    errors = ""
    if request.method == "POST":
        try:
            session["position_name"] = request.form["position_name"]
            session["inputs"].append(session["position_name"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["position_name"])
        try:
            session["job_category"] = request.form["job_category"]
            session["inputs"].append(session["job_category"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["job_category"])
        try:
            session["salaryL"] = request.form["salaryL"] #change to float?
            session["inputs"].append(session["salaryL"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["salaryL"])
        try:
            session["salaryU"] = request.form["salaryU"] #change to float?
            session["inputs"].append(session["salaryU"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["salaryU"])
        try:
            session["type_job"] = request.form["type_job"]
            session["inputs"].append(session["type_job"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["type_job"])
        try:
            session["shift_schedule"] = request.form["shift_schedule"]
            session["inputs"].append(session["shift_schedule"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["shift_schedule"])
        try:
            session["location"] = request.form["location"]
            session["inputs"].append(session["location"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["location"])

        if request.form["action"] == "Process Query":
            valid_query = True
            for i in session["inputs"]:
                if i == "":
                    valid_query = False
            if valid_query:
                result = process(session["inputs"])
                session["inputs"].clear()
                session.modified = True
                return '''
                    <html>
                        <body>
                            <p>{result}</p>
                            <p><a href="/">Click here to calculate again</a>
                        </body>
                    </html>
                '''.format(result=result)
            else:
                session["inputs"].clear()
                session.modified = True
                return '''
                    <html>
                        <body>
                            <p>Invalid Query</p>
                            <p><a href="/">Click to query again.</a>
                        </body>
                    </html>
                '''
        if request.form["action"] == "Start Over":
            session["inputs"].clear()
            session.modified = True

    return '''
        <html>
            <body>
                {errors}
                <p>Enter your desired job information:</p>
                <form method="post" action=".">
                    <p>Position Name:</p>
                    <p><input name="position_name" /></p>
                    <p>Job Category:</p>
                    <p><input name="job_category" /></p>
                    <p>Salary Lower Bound (in 1,000s of USD):</p>
                    <p><input name="salaryL" /></p>
                    <p>Salary Upper Bound (in 1,000s of USD):</p>
                    <p><input name="salaryU" /></p>
                    <p>Type of Job (Full Time or Part Time):</p>
                    <p><input name="type_job" /></p>
                    <p>Shift Schedule(e.g.Monday-Friday):</p>
                    <p><input name="shift_schedule" /></p>
                    <p>Location:</p>
                    <p><input name="location" /></p>
                    <p><input type="submit" name="action" value="Start over" /></p>
                    <p><input type="submit" name="action" value="Process Query" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)

    #return 'shiny Flash app for ling472/anly521 final project!'

