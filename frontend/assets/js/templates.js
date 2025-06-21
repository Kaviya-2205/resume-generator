// Handles template switching in preview.html

document.addEventListener("DOMContentLoaded", function () {
  const resumeData = JSON.parse(localStorage.getItem("resumeData"));
  const preview = document.getElementById("resume-preview");

  if (!resumeData || !preview) return;

  const { name, phone, email, skills, projects, certificates, objective, short_description, template_style } = resumeData;

  const formattedSkills = skills.join(", ");
  const formattedProjects = projects.join(", ");
  const formattedCertificates = certificates.join(", ");

  let content = `
    <h2>${name}</h2>
    <p><strong>Email:</strong> ${email}</p>
    <p><strong>Phone:</strong> ${phone}</p>
    <p><strong>Objective:</strong> ${objective}</p>
    <p><strong>Summary:</strong> ${short_description}</p>
    <p><strong>Skills:</strong> ${formattedSkills}</p>
    <p><strong>Projects:</strong> ${formattedProjects}</p>
    <p><strong>Certificates:</strong> ${formattedCertificates}</p>
  `;

  // Optionally change styles/layout based on template_style (just dummy logic for now)
  if (template_style === "modern") {
    preview.style.borderLeft = "5px solid #00ffff";
    preview.style.paddingLeft = "15px";
  } else if (template_style === "professional") {
    preview.style.backgroundColor = "#1c1c1c";
    preview.style.fontFamily = "Georgia, serif";
  }

  preview.innerHTML = content;
});