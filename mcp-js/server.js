require('dotenv').config();
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const sharp = require('sharp');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');

const { parsePrompt } = require('./services/geminiText');
const { editImage } = require('./services/geminiImage');

const PORT = process.env.PORT || 3000;
const STORAGE_DIR = process.env.STORAGE_DIR || './storage';
const MAX_FILE_MB = parseInt(process.env.MAX_FILE_MB || '8', 10);

if (!fs.existsSync(STORAGE_DIR)) fs.mkdirSync(STORAGE_DIR, { recursive: true });

const app = express();
app.use(express.json());
app.use(cors());
app.use(helmet());
app.use(rateLimit({ windowMs: 60 * 1000, max: 60 }));

const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, STORAGE_DIR),
    filename: (req, file, cb) => `${Date.now()}-${uuidv4()}${path.extname(file.originalname)}`
  }),
  limits: { fileSize: MAX_FILE_MB * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = /jpeg|jpg|png|webp/.test(file.mimetype) || /\.(jpe?g|png|webp)$/i.test(file.originalname);
    cb(allowed ? null : new Error('Only images are allowed (jpg/png/webp).'), allowed);
  }
});

app.get('/health', (req, res) => res.json({ status: 'ok', ts: Date.now() }));

app.post('/process', upload.single('image'), async (req, res) => {
  try {
    const file = req.file;
    const prompt = (req.body.prompt || '').trim();

    if (!prompt || !file) {
      if (file) fs.unlinkSync(file.path);
      return res.status(400).json({ error: 'Prompt and image are required.' });
    }

    const previewPath = path.join(STORAGE_DIR, `preview-${path.basename(file.path)}.webp`);
    await sharp(file.path).resize({ width: 1024, withoutEnlargement: true }).toFile(previewPath);

    // 1️⃣ Parse prompt with Gemini Text service
    const parsed = await parsePrompt(prompt);

    // 2️⃣ Edit image with Gemini Image service
    const out = await editImage(file.path, parsed.text);

    const outFilename = `out-${Date.now()}-${uuidv4()}.png`;
    const outPath = path.join(STORAGE_DIR, outFilename);
    const buffer = Buffer.from(out.imageBase64, 'base64');
    fs.writeFileSync(outPath, buffer);

    res.json({
      success: true,
      parsed: parsed.text,
      imageUrl: `/storage/${outFilename}`,
      previewUrl: `/storage/${path.basename(previewPath)}`
    });

  } catch (err) {
    console.error('Processing error:', err);
    res.status(500).json({ error: 'Server error', message: err.message || String(err) });
  }
});

app.use('/storage', express.static(path.resolve(STORAGE_DIR)));

app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
