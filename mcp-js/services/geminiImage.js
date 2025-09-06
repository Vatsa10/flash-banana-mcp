const { GoogleGenAI } = require('@google/genai');
const fs = require('fs');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function editImage(imagePath, instructions) {
  try {
    const resp = await ai.models.generateImage({
      model: 'gemini-2.5-flash-image',
      contents: instructions,
      image: fs.createReadStream(imagePath)
    });
    return resp;
  } catch (err) {
    console.error('Gemini Image editing failed:', err);
    return { imageBase64: null, error: err.message };
  }
}

module.exports = { editImage };
