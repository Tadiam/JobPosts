from flask import Flask, request, session
from processing import process

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "93e78914d7b162b36641ca4ec562c86d4f14a399e2629ca218742fbc0610168e"

@app.route("/", methods = ["GET", "POST"])
def hello_world():
    if "inputs" not in session:
        session["inputs"] = []
        session["description"] = ""
    errors = ""
    if request.method == "POST":
        try:
            session["description"] = request.form["description"]
            session["inputs"].append(session["description"])
            session.modified = True
        except:
            errors += "<p>{!r} is invalid.</p>\n".format(request.form["position_name"])

        if request.form["action"] == "Process Query":
            valid_query = True
            
            if valid_query:
                result = process(session["inputs"])
                session["inputs"].clear()
                session.modified = True
                return '''
                    <html>
                        <body>
                            <p>{result}</p>
                            <p><a href="/">Click here to query again</a>
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
                <p>Enter a description of your qualifications and job preferences, about 100 words, below.</p>
                <p>Example:</p>
                <p>Work on projects in data mining and knowledge discovery. Past experience with simulation models and visualization.</p>
                <p>Highly motivated worker, enjoys collaborative work environments. I have my Masters in Data Science and Analytics.</p>
                <form method="post" action=".">
                    <p><input name="description" /></p>
                    <p><input type="submit" name="action" value="Start over" /></p>
                    <p><input type="submit" name="action" value="Process Query" /></p>
                </form>
            </body>
        </html>
    '''.format(errors=errors)

    #return 'shiny Flash app for ling472/anly521 final project!'


