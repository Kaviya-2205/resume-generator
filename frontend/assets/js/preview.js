document.addEventListener("DOMContentLoaded", function () {
  const resumeContainer = document.getElementById("resume-preview");
  const downloadBtn = document.getElementById("download-btn");

  const resumeData = JSON.parse(localStorage.getItem("resumeData"));

  if (!resumeData) {
    resumeContainer.innerHTML = "<p>No resume data found. Please go back and fill the form.</p>";
    return;
  }

  const {
    name,
    phone,
    email,
    skills,
    projects,
    certificates,
    job_description,
    objective,
    short_description,
    template_style
  } = resumeData;

  resumeContainer.innerHTML = `
    <div class="resume-card">
      <h2>${name}</h2>
      <p><strong>Email:</strong> ${email}</p>
      <p><strong>Phone:</strong> ${phone}</p>
      <p><strong>Objective:</strong> ${objective}</p>
      <p><strong>Short Description:</strong> ${short_description}</p>
      <hr/>
      <p><strong>Skills:</strong> ${skills.join(", ")}</p>
      <p><strong>Projects:</strong> ${projects.join(", ")}</p>
      <p><strong>Certificates:</strong> ${certificates.join(", ")}</p>
      <p><strong>Job Description:</strong> ${job_description}</p>
      <p><strong>Template Style:</strong> ${template_style}</p>
    </div>
  `;

  downloadBtn.addEventListener("click", () => {
    html2pdf()
      .from(resumeContainer)
      .set({ margin: 0.5, filename: "resume.pdf", image: { type: "jpeg", quality: 0.98 }, html2canvas: { scale: 2 }, jsPDF: { unit: "in", format: "letter", orientation: "portrait" } })
      .save();
  });
});