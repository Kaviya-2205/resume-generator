from flask import Flask, render_template_string, request
import pandas as pd
import random

app = Flask(__name__)

# Load data
df = pd.read_csv("resume_data.csv")

# HTML template
html = """
<!doctype html>
<title>Resume Generator</title>
<h2 style="text-align:center;">Simple AI Resume Generator</h2>
<form method="POST" style="max-width:600px;margin:auto;padding:20px;border:1px solid #ccc;border-radius:10px;">
  {% for field in ['name', 'email', 'phone', 'education', 'certification', 'skills', 'projects'] %}
    <label for="{{field}}">{{field.capitalize()}}:</label><br>
    <input type="text" name="{{field}}" style="width:100%;padding:10px;margin-bottom:10px;" required><br>
  {% endfor %}
  <input type="submit" value="Generate Resume" style="width:100%;padding:10px;background:#007bff;color:white;border:none;border-radius:5px;">
</form>

{% if result %}
  <div style="max-width:600px;margin:30px auto;padding:20px;border:2px solid #4CAF50;border-radius:10px;background:#f9f9f9;">
    <h3>Generated Resume</h3>
    <p><strong>Name:</strong> {{result.name}}</p>
    <p><strong>Email:</strong> {{result.email}}</p>
    <p><strong>Phone:</strong> {{result.phone}}</p>
    <p><strong>Education:</strong> {{result.education}}</p>
    <p><strong>Certifications:</strong> {{result.certification}}</p>
    <p><strong>Skills:</strong> {{result.skills}}</p>
    <p><strong>Projects:</strong> {{result.projects}}</p>
    <p><strong>Objective/Summary:</strong><br>{{result.generated_resume}}</p>
  </div>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        # Get user input
        user_input = {
            "name": request.form["name"],
            "email": request.form["email"],
            "phone": request.form["phone"],
            "education": request.form["education"],
            "certification": request.form["certification"],
            "skills": request.form["skills"],
            "projects": request.form["projects"],
        }

        # Simulate AI by picking similar entry from dataset
        result_row = df.sample(1).iloc[0]
        user_input["generated_resume"] = result_row["generated_resume"]
        result = type("Resume", (object,), user_input)

    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)