const { GoogleGenAI } = require('@google/genai');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function parsePrompt(prompt) {
  try {
    const resp = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt
    });
    return resp;
  } catch (err) {
    console.error('Gemini Text parsing failed:', err);
    return { text: prompt, error: err.message };
  }
}

module.exports = { parsePrompt };
