document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("resumeForm");

  form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const formData = new FormData(form);
    const payload = {
  name: formData.get("name"),
  phone: formData.get("phone"),
  email: formData.get("email"),
  skills: formData.get("skills").split(",").map(s => s.trim()),
  projects: formData.get("projects").split(",").map(p => p.trim()),
  certificates: formData.get("certificates").split(",").map(c => c.trim()),
  job_description: formData.get("job_description"),
  short_resume_text: "Motivated learner and quick adopter."  // ✅ Add this
  };
    try {
      const response = await fetch("http://127.0.0.1:8000/api/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) throw new Error("Generation failed");

      const data = await response.json();

      // Store in localStorage
      localStorage.setItem("resumeData", JSON.stringify({
        ...payload,
        objective: data.objective,
        short_description: data.short_description
      }));

      // ✅ Redirect to preview page
      window.location.href = "/preview";
    } catch (err) {
      console.error(err);
      alert("Something went wrong!");
    }
  });
});