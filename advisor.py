# ==========================================
# 4. GROQ ADVISOR ENGINE
# ==========================================
def _get_groq_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or Streamlit secrets.")
    return Groq(api_key=key)


def generate_advice(history):
    """Summarizes engagement metrics and prompts Groq Llama 3.3 for pedagogical advice."""
    if not history:
        return "No data collected yet."

    total_snapshots = len(history)
    student_ids = sorted({s["student_id"] for snap in history for s in snap})

    summary = {}
    for sid in student_ids:
        engaged = []
        ears = []
        for snap in history:
            for s in snap:
                if s["student_id"] == sid:
                    engaged.append(1 if s["engaged"] else 0)
                    ears.append(s["ear"])
        if engaged:
            summary[sid] = {
                "engagement_rate": round(sum(engaged) / len(engaged), 2),
                "avg_ear": round(sum(ears) / len(ears), 3),
                "appearances": len(engaged),
            }

    prompt = f"""You are an educational consultant analysing anonymised classroom engagement data.

Total snapshots: {total_snapshots}
Per-student summary (engagement_rate is 0.0-1.0; avg_ear below 0.18 suggests drowsiness):
{json.dumps(summary, indent=2)}

Tasks:
1.⁠ ⁠Identify any student with engagement_rate < 0.5 or avg_ear < 0.18.
2.⁠ ⁠Suggest ONE concrete pedagogical adjustment the teacher can make in the next 10 minutes.
Keep the response under 150 words. Be specific, encouraging, and actionable.
"""

    client = _get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a concise, practical teaching assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content
