import React, { useState } from "react";

function App() {
  const [resumeFile, setResumeFile] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setResumeFile(e.target.files[0]);
  };

  const handleJDChange = (e) => {
    setJobDescription(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("resume_file", resumeFile);
    formData.append("job_description", jobDescription);

    try {
      const response = await fetch("http://localhost:8000/analyze/", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: "Failed to analyze resume." });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 600, margin: "auto", padding: 20 }}>
      <h2>Resume Relevance Checker</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Upload Resume (PDF): </label>
          <input type="file" accept=".pdf" onChange={handleFileChange} required />
        </div>
        <div style={{ marginTop: 10 }}>
          <label>Job Description:</label>
          <textarea
            rows={5}
            style={{ width: "100%" }}
            value={jobDescription}
            onChange={handleJDChange}
            required
          />
        </div>
        <button type="submit" disabled={loading} style={{ marginTop: 10 }}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>
      {result && (
        <div style={{ marginTop: 20 }}>
          {result.error ? (
            <div style={{ color: "red" }}>{result.error}</div>
          ) : (
            <>
              <h4>Relevance Score: {result.relevance_score}%</h4>
              <h5>Extracted Resume Text:</h5>
              <pre style={{ background: "#f4f4f4", padding: 10 }}>{result.extracted_resume_text}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;